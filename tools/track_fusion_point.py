import argparse
import copy
import os
import sys
import time
import json
import math
import numpy as np
import shutil
import tf

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy
from utils.utils import log_parser_args
from utils.box_utils import box2d_filter, get_3d_box_8corner, get_3d_box_2corner
from utils.box_utils import nms

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]

# 99.9 percentile of the l2 velocity error distribution (per class / 0.5 second)
# This is an earlier statistics and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car': 3,
    'truck': 4,
    'bus': 5.5,
    'trailer': 2,
    'pedestrian': 1,
    'motorcycle': 4,
    'bicycle': 2.5,
    'construction_vehicle': 1,
    'barrier': 1,
    'traffic_cone': 1,
}


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

# DDFish tracking usage ==========================================
def q_to_xyzw(Q):
        '''
        wxyz -> xyzw
        '''
        return [Q[1], Q[2], Q[3], Q[0]]

def q_to_wxyz(Q):
    '''
    xyzw -> wxyz
    '''
    return [Q[3], Q[0], Q[1], Q[2]]

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)  (w,x,y,z)
    
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
        
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
        
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
        
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

class nusc_dataset:
    def __init__(
        self, 
        nusc_version='v1.0-trainval', 
        split='val', 
        nusc_path='/data/nuscenes',
        result_path='data/detection_result.json',
        frame_meta_path='data/frames_meta.json',
        radar_pc_path='data/radar_PC/radar_PC_13Hz_with_vcomp.json',
    ):
        # self.nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
        self.det_res = self.load_detections(result_path)
        self.frames = self.load_frames_meta(frame_meta_path)
        self.radar_PCs = self.load_radar_pc(radar_pc_path)
        self.key_radar_PCs = self.load_key_radar_pc()
        if split == 'train':
            self.scene_names = splits.train
        elif split == 'val':
            self.scene_names = splits.val
        elif split == 'test':
            sys.exit("Not support test data yet!")
        else:
            sys.exit(f"No split type {split}!")

    def load_detections(self, path):
        with open(path, 'rb') as f:
            detections = json.load(f)
        return detections

    def load_frames_meta(self, path):
        with open(path, 'rb') as f:
            frames = json.load(f)
        return frames
    
    def load_radar_pc(self, path):
        if path is None or path == "None":
            print(f"radar pointcloud path is None!")
            return
        with open(path, 'rb') as f:
            radar_PCs = json.load(f)['radar_PCs']
        return radar_PCs
    
    def load_key_radar_pc(self):
        count = 0
        key_radar_PCs = {}
        for k, radar_pc in self.radar_PCs.items():
            if not radar_pc['is_key_frame']:
                continue
            sample_token = radar_pc['sample_token']
            key_radar_pc = {
                'token': sample_token,
                'ego_pose_token': radar_pc['ego_pose_token'],
                'points': radar_pc['points'],
            }
            key_radar_PCs.update({sample_token: key_radar_pc})
            count += 1
        print(f"{count} radar_PCs loaded")
        return key_radar_PCs

    def get_det_meta(self):
        return self.det_res['meta']

    def get_det_results(self, bbox_th):
        return self.filter_box(self.det_res['results'], bbox_th)

    def get_frames_meta(self):
        return self.frames['frames']
    
    def get_key_radar_pc(self, key_token):
        return self.key_radar_PCs[key_token]['points']

    def get_4f_transform(self, pose, inverse=False):
        return transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=inverse)

    def lidar2world(self, objects, token, inverse=False):
        '''
        Transform objects from lidar coordinates to world coordinates
        '''
        objects = deepcopy(objects)

        sample_record = self.nusc.get('sample', token)
        LIDAR_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', LIDAR_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', LIDAR_record['calibrated_sensor_token'])
        lidar2car = self.get_4f_transform(cs_record, inverse=inverse)
        car2world = self.get_4f_transform(ego_pose, inverse=inverse)

        ret = []
        for object in objects:
            trans = np.array(object['translation'])
            vel = np.array([object['velocity'][0], object['velocity'][1], 0.0])
            rot = quaternion_rotation_matrix(object['rotation'])
            trans = np.hstack([rot, trans.reshape(-1, 1)])
            trans = np.vstack([trans, np.array([0, 0, 0, 1])]).reshape(-1, 4)
            vel = vel.reshape(-1, 1)
            if not inverse:
                new_trans = car2world.dot(lidar2car.dot(trans))
                new_vel = car2world[:3, :3].dot(lidar2car[:3, :3].dot(vel))
            elif inverse:
                new_trans = lidar2car.dot(car2world.dot(trans))
                new_vel = lidar2car[:3, :3].dot(car2world[:3, :3].dot(vel))
            object['translation'] = new_trans[:3, 3].ravel().tolist()
            object['rotation'] = q_to_wxyz(R.from_matrix(new_trans[:3, :3]).as_quat())
            object['velocity'] = new_vel.ravel()[:2]
            ret.append(object)

        return ret

    def filter_box(self, det_res, th):
        print("======")
        print(f"Filtering bboxes by threshold {th}...", end='')
        ret_dict = {}
        for i, (token, bboxes) in enumerate(det_res.items()):
            ret_bboxes = []
            for bbox in bboxes:
                if bbox['detection_score'] < th:
                    continue
                ret_bboxes.append(bbox)
            ret_dict.update({token: ret_bboxes})
        print("Done.")
        return ret_dict

''' ================================================================== '''
''' ====================== Radar tracking usage ====================== '''
''' ================================================================== '''
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
RADAR_NAMES = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)

def comparing_positions(self, positions1_data, positions2_data, positions1, positions2):
    M = len(positions1_data)
    N = len(positions2_data)

    # Set the considered range threshold
    positions1_init = (np.array([np.sqrt(point['velocity'][0]**2 + point['velocity'][1]**2) for point in positions1_data], np.float32) >= 1.5) + 0.5   # M pos1 saw first time
    # Max distance
    max_diff = np.array(3.5 * positions1_init, np.float32)

    if len(positions1) > 0:  # NOT FIRST FRAME
        dist = (((positions1.reshape(1, -1, 2) - positions2.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
        dist = np.sqrt(dist)  # absolute distance in meter
        invalid = (dist > max_diff.reshape(1, M)) > 0
        dist = dist + invalid * 1e18
        if self.hungarian:
            dist[dist > 1e18] = 1e18
            matched_indices = linear_sum_assignment(deepcopy(dist))
        else:
            matched_indices = greedy_assignment(deepcopy(dist))
    else:  # first few frame
        assert M == 0
        matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_positions1_data = [d for d in range(positions1.shape[0]) if not (d in matched_indices[:, 1])]
    unmatched_positions2_data = [d for d in range(positions2.shape[0]) if not (d in matched_indices[:, 0])]

    if self.hungarian:
        matches = []
        for m in matched_indices:
            if dist[m[0], m[1]] > 1e16:
                unmatched_positions2_data.append(m[0])
            else:
                matches.append(m)
        matches = np.array(matches).reshape(-1, 2)
    else:
        matches = matched_indices
    return matches, unmatched_positions1_data, unmatched_positions2_data

class RadarTracker(object):
    def __init__(self, tracker, max_age=6, min_hits=1, hungarian=False, use_vel=False):
        # Tracker should be 'KF' or 'PointTracker'
        self.tracker = tracker
        self.hungarian = hungarian
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_vel = use_vel
        self.id_count = 0
        self.tracks = []
        self.reset()
        print(f"Use radar velocity: {self.use_vel}")

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def step_centertrack(self, radar_points, time_lag):
        """
        computes connections between current resources with resources from older frames
        :param radar_points: radar pointclouds from frames
        :param time_lag: time between two frame (difference in their timestamp)
        :return: tracks: tracklets (detection + tracking id, age, activity) for one specific frame
                 if train_data true than also return the training data
        """
        # if no radar points in this frame, reset tracks list and return empty list
        if len(radar_points) == 0:
            self.tracks = []  # <-- however, this means, all tracklets are gone (i.e. 'died')
            return []
        # --> radar_points[i]: {'pose':[x, y], 'vel':mirror_velocity, 'vel_comp':[vx, vy]}
        else:
            temp = []
            for point in radar_points:
                if point['vel'] > 0.5:
                    point['ct'] = np.array(point['pose'])  # ct: 2d centerpoint of one detection
                    if self.tracker == 'PointTracker':
                        if time_lag != 0:
                            point['tracking'] = np.array(point['velocity'][:2]) * -1 * time_lag
                        else:
                            point['tracking'] = np.array([0.0, 0.0])
                    temp.append(point)
            radar_points = temp

        N = len(radar_points)  # number of radar points in this frame
        M = len(self.tracks)  # number of tracklets
        ret = []  # initiate return value (will become the updated tracklets list)

        # if no tracklet exist just yet (i.e. processing the first frame)
        if M == 0:
            for point in radar_points:  # for each (extended) detection
                # initiate new tracklet
                track = point
                self.id_count += 1
                # extend tracklet with the following attributes:
                track['tracking_id'] = self.id_count  # tracklet id
                track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
                track['active'] = self.min_hits  # currently matched? (start with 1)
                # track['velocity'] = [0.0, 0.0]
                track['velocity'] = point['vel_comp']
                if self.tracker == 'KF':
                    if self.use_vel:
                        track['KF'] = KalmanFilter(6, 4)
                        track['KF'].H = np.array([[1., 0., 0., 0., 0., 0.],
                                                  [0., 1., 0., 0., 0., 0.],
                                                  [0., 0., 1., 0., 0., 0.],
                                                  [0., 0., 0., 1., 0., 0.]])
                    else:
                        track['KF'] = KalmanFilter(6, 2)
                        track['KF'].H = np.array([[1., 0., 0., 0., 0., 0.],
                                                  [0., 1., 0., 0., 0., 0.]])
                    track['KF'].x = np.hstack([track['ct'], track['velocity'], np.zeros(2)])
                    track['KF'].F = np.array([[1, 0, time_lag, 0, 0.5 * time_lag * time_lag, 0],
                                                [0, 1, 0, time_lag, 0, 0.5 * time_lag * time_lag],
                                                [0, 0, 1, 0, time_lag, 0],
                                                [0, 0, 0, 1, 0, time_lag],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]])
                    track['KF'].P *= 5
                    track['KF'].R *= 1
                    track['KF'].Q *= 1
                ret.append(track)
            self.tracks = ret
            return ret

        # Processing from the second frame
        if self.tracker == 'PointTracker':
            # N x 2
            # dets: estmated 2d centerpoint of a detection in the previous frame (ct + expected offset)
            if N > 0:
                if 'tracking' in radar_points[0]:
                    dets = np.array(
                        [det['ct'].astype(np.float32) + det['tracking'].astype(np.float32)
                        for det in radar_points], np.float32)
                else:
                    dets = np.array(
                        [det['ct'] for det in radar_points], np.float32)
            else:
                dets = np.array([], np.float32)

            tracks = np.array(
                [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        elif self.tracker == 'KF':
            if N > 0:
                dets = np.array(
                    [det['ct'] for det in radar_points], np.float32)
            else:
                dets = np.array([], np.float32)

            tracks = []
            for tracklet in self.tracks:
                tracklet['KF'].predict(F=np.array([[1, 0, time_lag, 0, 0.5 * time_lag * time_lag, 0],
                                                   [0, 1, 0, time_lag, 0, 0.5 * time_lag * time_lag],
                                                   [0, 0, 1, 0, time_lag, 0],
                                                   [0, 0, 0, 1, 0, time_lag],
                                                   [0, 0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 0, 1]]))
                tracks.append(tracklet['KF'].x[:2])

            tracks = np.array(tracks, np.float32)  # M x 2

        # matching the current with the estimated pass
        matching = comparing_positions(self, self.tracks, radar_points, tracks, dets)
        matched, unmatched_trk, unmatched_det = matching[0], matching[1], matching[2]
        # print("Matched:", matched)
        # exit(0)

        # add matches
        for m in matched:
            # initiate new tracklet (with three additional attributes)
            track = radar_points[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']  # tracklet id = id of matched trackled
            track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
            track['active'] = self.tracks[m[1]]['active'] + 1
            if self.tracker == 'KF':
                track['KF'] = self.tracks[m[1]]['KF']
                if self.use_vel:
                    # print("Update (Matched velocity): ", np.array((track['ct'] - self.tracks[m[1]]['ct']) / time_lag))
                    track['KF'].update(z=np.hstack([track['ct'], np.array((track['ct'] - self.tracks[m[1]]['ct']) / time_lag)]))
                else:
                    track['KF'].update(z=track['ct'])
                # More stable tracking
                if track['active'] == 5:
                    track['KF'].Q *= 0.1
                if track['active'] == 8:
                    track['KF'].Q *= 0.1
                if track['active'] == 11:
                    track['KF'].Q *= 0.5
                track['ct'][0] = track['KF'].x[0]
                track['ct'][1] = track['KF'].x[1]
                track['velocity'] = list(track['KF'].x[2:4])

            ret.append(track)

        # add unmatched resources as new 'born' tracklets
        for i in unmatched_det:
            track = radar_points[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            # track['velocity'] = [0.0, 0.0]
            track['velocity'] = track['vel_comp']
            if self.tracker == 'KF':
                if self.use_vel:
                    track['KF'] = KalmanFilter(6, 4)
                    track['KF'].H = np.array([[1., 0., 0., 0., 0., 0.],
                                              [0., 1., 0., 0., 0., 0.],
                                              [0., 0., 1., 0., 0., 0.],
                                              [0., 0., 0., 1., 0., 0.]])
                else:
                    track['KF'] = KalmanFilter(6, 2)
                    track['KF'].H = np.array([[1., 0., 0., 0., 0., 0.],
                                              [0., 1., 0., 0., 0., 0.]])
                track['KF'].x = np.hstack([track['ct'], track['velocity'], np.zeros(2)])
                track['KF'].P *= 50
                track['KF'].R *= 1
                track['KF'].Q *= 1

            ret.append(track)

        # still store unmatched tracks, however, we shouldn't output the object in current frame
        for i in unmatched_trk:
            track = self.tracks[i]

            # keep tracklet if score is above threshold AND age is not too high
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']

                if 'tracking' in track:
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                    track['velocity'] = list(offset)
                elif self.tracker == 'KF':
                    track['ct'] = track['KF'].x[:2]
                    track['velocity'] = list(track['KF'].x[2:4])

                track['pose'] = list(track['ct'])

                ret.append(track)

        for track in ret:
            track['pose'] = list(track['ct'])
            track['translation'] = list(track['ct'])

        self.tracks = ret
        return ret

def shift_quaternion(Q):
    """
    Convert (w x y z) to (x y z w) format
    Input Q: ndarray (1, 4)
    Output : ndarray (1, 4)
    """
    return np.concatenate((np.array(Q)[1:4], np.array(Q)[0:1]))

def get_yaw_from_quaternion(Q):
    Quaternion = shift_quaternion(Q)
    euler = tf.transformations.euler_from_quaternion(Quaternion)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    return yaw

def vel_project(vel, target_vel):
    '''
    Project vel to target_vel (2D)
    '''
    vel = np.array(vel)
    t_vel = np.array(target_vel)
    new_vel = vel @ t_vel.T / (t_vel[0]**2 + t_vel[1]**2) * t_vel
    # mag = np.sqrt(vel[0]**2 + vel[1]**2) / np.sqrt(new_vel[0]**2 + new_vel[1]**2)
    # new_vel = [new_vel[0] * mag, new_vel[1] * mag]
    return list(new_vel)

import matplotlib.pyplot as plt
def vel_fusion(dets, radar_pc):
    r_points = [point['pose'] for point in radar_pc]
    for det in dets:
        trans = det['translation']
        size = deepcopy(det['size'])
        size = [size[1] + 1.0, size[0] + 1.0, size[2]]
        Q = det['rotation']
        heading = get_yaw_from_quaternion(Q)
        corners = get_3d_box_2corner(trans, size, heading)
        corners = list(corners[0][:2]) + list(corners[1][:2])
        inbox, outbox, idxs = box2d_filter(corners, r_points)
        for idx in idxs:
            radar_pc[idx]['vel_comp'] = vel_project(radar_pc[idx]['vel_comp'], det['velocity'])

    # plt.figure(figsize=(12, 8))
    # plt.clf()
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')
    # for point in radar_pc:
    #     plt.scatter(point['pose'][0], point['pose'][1], s=5)
    #     plt.plot([point['pose'][0], point['pose'][0]+point['vel_comp'][0]],
    #              [point['pose'][1], point['pose'][1]+point['vel_comp'][1]])
    # for det in dets:
    #     size = deepcopy(det['size'])
    #     size = [size[1], size[0], size[2]]
    #     corners3d = get_3d_box_8corner(det['translation'], size, get_yaw_from_quaternion(det['rotation']))
    #     corners3d = corners3d.transpose(1,0).ravel()
    #     corners2d = np.concatenate([
    #         np.array([corners3d[0:4], corners3d[8:12]]), 
    #         np.array([[corners3d[0]], [corners3d[8]]])], axis=1)
    #     plt.plot(corners2d[0], corners2d[1])
    # plt.show()
    return radar_pc

# Tracking usage ==========================================

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="data/track_results")

    parser.add_argument("--tracker", type=str, default='PointTracker')
    parser.add_argument("--min_hits", type=int, default=1)
    parser.add_argument("--max_age", type=int, default=6)
    parser.add_argument("--det_th", type=float, default=0.0)
    parser.add_argument("--del_th", type=float, default=0.0)
    parser.add_argument("--active_th", type=float, default=1.0)
    parser.add_argument("--update_function", type=str, default=None)
    parser.add_argument("--score_decay", type=float, default=0.15)
    parser.add_argument("--use_nms", type=int, default=0)
    parser.add_argument("--nms_th", type=float, default=0.5)
    parser.add_argument("--use_vel", type=int, default=1)
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--dataroot", type=str, default='data/nuscenes')
    parser.add_argument("--workspace", type=str, default='/home/Student/Tracking')
    parser.add_argument("--detection_path", type=str, default='data/detection_result.json')
    parser.add_argument("--frames_meta_path", type=str, default='data/frames_meta.json')
    parser.add_argument("--radar_fusion", type=int, default=1)
    parser.add_argument("--radar_tracker_path", type=str, default=None)
    parser.add_argument("--radar_pc_path", type=str, default='data/radar_PC/radar_PC_13Hz_with_vcomp.json')
    args, opts = parser.parse_known_args()

    # Notice
    if args.radar_fusion:
        if args.radar_tracker_path is None:
            print("Warning: args.radar_tracker_path is None!")

    # Prepare results saving path and save parameters
    root_path = args.out_dir
    track_res_path = os.path.join(root_path, 'tracking_result.json')
    mkdir_or_exist(os.path.dirname(track_res_path))
    log_parser_args(root_path, args)
    shutil.copyfile(os.path.join(args.workspace, 'tools/track_fusion_point.sh'), os.path.join(root_path, 'track.sh'), follow_symlinks=True)

    # Build data preprocessing for track
    dataset = nusc_dataset(
        nusc_version='v1.0-trainval',
        split=args.split,
        nusc_path=args.dataroot,
        result_path=args.detection_path,
        frame_meta_path=args.frames_meta_path,
        radar_pc_path=args.radar_pc_path,
    )

    # from tracker import PubTracker
    from tracker_radar_point import PubTracker, RADAR_FUSION_NAMES
    tracker = PubTracker(
        hungarian=False,
        max_age=args.max_age,
        active_th=args.active_th,
        min_hits=args.min_hits,
        update_function=args.update_function,
        score_decay=args.score_decay,
        deletion_th=args.del_th,
        detection_th=args.det_th,
        use_vel=args.use_vel,
        tracker=args.tracker,
        radar_fusion=args.radar_fusion,
    )
    # prepare writen output file
    nusc_annos_trk = {
        "results": {},
        "meta": None,
    }

    # prepare tracker
    radar_tracker = RadarTracker(
        tracker="KF", 
        max_age=6, 
        min_hits=1, 
        hungarian=False, 
        use_vel=False
    )
    # prepare writen output file
    radar_points_trk = {
        "key_results": {}, 
        "meta": None,
    }
    
    # Load detection results
    detections = dataset.get_det_results(args.bbox_score)
    frames = dataset.get_frames_meta()
    len_frames = len(frames)
    prev_activate_num = 0
    radar_active_frames = {}

    # start tracking *****************************************
    print("Begin Tracking\n")
    start = time.time()

    for i in tqdm(range(len_frames)):
        # get frameID (=token)
        token = frames[i]['token']
        timestamp = frames[i]['timestamp']

        name = "{}-{}".format(timestamp, token)

        # Reset tracker if this is first frame of the sequence
        if frames[i]['first']:
            tracker.reset()
            radar_tracker.reset()
            last_time_stamp = timestamp

        # calculate time between two frames
        time_lag = (timestamp - last_time_stamp)
        last_time_stamp = timestamp

        # get tracklets of current frame
        det = detections[token]

        if args.use_nms:
            det, _ = nms(det, args.nms_th)

        if args.radar_fusion:
            radar_pc = dataset.get_key_radar_pc(token)
            # Transform list of [x, y, z, vx, vy] to list of {'pose':[x, y], 'vel':mirror_velocity, 'vel_comp':[vx, vy]}
            inputs = []
            for point in radar_pc:
                inputs.append({
                    'pose': point[:2],
                    'translation': point[:2],
                    'vel': np.sqrt(point[3]**2 + point[4]**2),
                    'vel_comp': point[3:5],
                })
        else:
            inputs = []

        outputs = tracker.step_centertrack(det, time_lag, inputs)

        for item in outputs:
            item['tracking_score'] = item['detection_score']

        # prepare writen results file
        annos_trk = []

        # Save tracking results in world coordinates
        for item in outputs:
            if 'active' in item and item['active'] < args.min_hits:
                continue

            if item['detection_name'] in NUSCENES_TRACKING_NAMES:
                nusc_trk = {
                    "sample_token": token,
                    "translation": list(item['translation']),
                    "size": list(item['size']),
                    "rotation": list(item['rotation']),
                    "velocity": list(item['velocity']),
                    "tracking_id": str(item['tracking_id']),
                    "tracking_name": item['detection_name'],
                    "tracking_score": item['tracking_score'],
                }
                annos_trk.append(nusc_trk)

        nusc_annos_trk["results"].update({token: annos_trk})

        # if args.radar_fusion:
        # # Save radar tracking results
        #     points_trk = []
        #     for item in radar_tracker_result:
        #         if 'active' in item and item['active'] < args.min_hits:
        #             continue
        #         nusc_trk = {
        #             'sample_token': token,
        #             'translation': item['pose'],
        #             'velocity': item['velocity'],
        #             'tracking_id': str(item['tracking_id']),
        #         }
        #         points_trk.append(nusc_trk)
        #     radar_points_trk["key_results"].update({token: deepcopy(points_trk)})

        #     # Save Radar update frame info
        #     if args.radar_fusion:
        #         if prev_activate_num != tracker.Active_by_radarTracker:
        #             radar_active_frames[i] = token
        #             prev_activate_num = tracker.Active_by_radarTracker

    # calculate computation time
    end = time.time()
    second = (end - start)
    speed = len_frames / second
    print("======")
    print("The speed is {} FPS".format(speed))
    print("tracking results have {} frames".format(len(nusc_annos_trk["results"])))
    if args.radar_fusion:
        print(f"Radar update {len(radar_active_frames)} frames")
        print(f"Radar tracker update {tracker.Active_by_radarTracker} times")
        print("Radar tracker update objects:")
        for name, num in zip(RADAR_FUSION_NAMES, tracker.Active_by_radarTracker_cat):
            print(f"{name:20s}: {num:4d}")

    nusc_annos_trk["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    # Write radar update frames info
    if args.radar_fusion:
        radar_frames_path = os.path.join(root_path, "radar_update_frames.json")
        with open(radar_frames_path, "w") as f:
            json.dump(radar_active_frames, f, indent = 2)
        print(f"radar update frames info write to {radar_frames_path}")

    # write result file
    with open(track_res_path, "w") as f:
        json.dump(nusc_annos_trk, f)
    print(f"tracking results write to {track_res_path}\n")

    # # write radar result file
    # with open(os.path.join(root_path, 'radar_tracking_result_13Hz.json'), "w") as f:
    #     json.dump(radar_points_trk, f)

    # Evaluation
    if args.evaluate:
        print("======")
        print("Start evaluating tracking results")
        output_dir = os.path.join(root_path, 'eval')
        nusc_eval(
            track_res_path,
            "val",
            output_dir,  # instead of args.work_dir,
            args.dataroot
        )

    return speed


def nusc_eval(res_path, eval_set="val", output_dir=None, data_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs

    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=data_path,
    )
    metrics_summary = nusc_eval.main()

if __name__ == "__main__":
    main()
