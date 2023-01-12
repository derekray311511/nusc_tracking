import matplotlib.pyplot as plt
import json
import os
import time
import argparse
import numpy as np
import sys
import cv2

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import RadarPointCloud
from tqdm import tqdm
from geometry_utils import pc2world
from utils import mkdir_or_exist
from utils import get_current_datetime
from utils import Trajectory


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
                    track['KF'].x = np.hstack([track['ct'], np.zeros(2), np.zeros(2)])
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
                track['KF'].x = np.hstack([track['ct'], np.zeros(2), np.zeros(2)])
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

        self.tracks = ret
        return ret
        

def main(args, viz=False):
    print('Deploy OK')
    if viz:
        fig = plt.figure(figsize=(12, 8))
        def on_key(event):
            if event.key == 'escape':
                sys.exit(f"Exit by pressing {event.key}")

    # load radar pointclouds
    with open(os.path.join(args.out_dir, 'radar_PC_13Hz_with_vcomp.json'), 'rb') as f:
        PCs = json.load(f)
        meta = PCs['meta']
        radar_PCs = PCs['radar_PCs']

    # prepare writen output file
    radar_points_trk = {
        "results": {},
        "key_results": {}, 
        "meta": None,
    }

    # prepare tracker
    tracker = RadarTracker(
        tracker=args.tracker, 
        max_age=args.max_age, 
        min_hits=args.min_hits, 
        hungarian=args.hungarian, 
        use_vel=args.use_vel
    )

    # Open Nuscenes dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    scenes = splits.val

    # start tracking *****************************************
    print("Begin Tracking\n")
    start = time.time()

    numOfScene = len(nusc.scene)
    total_frame_num = 0
    key_frame_num = 0
    for scene_num in tqdm(range(numOfScene)):

        # Get first scene token (code number)
        my_scene = nusc.scene[scene_num]
        scene_name = my_scene['name']
        if scene_name not in scenes:
            continue

        # Load first sample data record
        sample_token = my_scene['first_sample_token']
        current_scene_name = scene_name
        sample = nusc.get('sample', sample_token)
        radar_sample_data = {
            'RADAR_FRONT': None, 
            'RADAR_FRONT_LEFT': None, 
            'RADAR_FRONT_RIGHT': None, 
            'RADAR_BACK_LEFT': None, 
            'RADAR_BACK_RIGHT': None,
        }
        for sensor_name in RADAR_NAMES:
            sample_data = nusc.get('sample_data', sample['data'][sensor_name])
            radar_sample_data.update({sensor_name: sample_data})
            last_timestamp = radar_sample_data[sensor_name]['timestamp']

        # Reset the tracker
        tracker.reset()

        # For visualization Trajectory
        if viz:
            radar_traj = Trajectory({})

        # Start tracking
        local_frame_num = 0
        while scene_name == current_scene_name:
            sample = nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            current_scene_name = nusc.get('scene', scene_token)['name']
            local_frame_num += 1

            # Get radar pointcloud and timestamp
            for sensor_name in RADAR_NAMES:
                if radar_sample_data[sensor_name] is None:
                    continue
                ego_pose_token = radar_sample_data[sensor_name]['ego_pose_token']
                radar_PC = radar_PCs[radar_sample_data[sensor_name]['token']]
                radar_pointcloud = radar_PC['points']
                timestamp = radar_sample_data[sensor_name]['timestamp']
                break

            # Transform list of [x, y, z, vx, vy] to list of {'pose':[x, y], 'vel':mirror_velocity, 'vel_comp':[vx, vy]}
            inputs = []
            for point in radar_pointcloud:
                inputs.append({
                    'pose': point[:2],
                    'vel': np.sqrt(point[3]**2 + point[4]**2),
                    'vel_comp': point[3:5],
                })

            # Track radar pointcloud
            time_lag = (timestamp - last_timestamp) * 1e-6
            last_timestamp = timestamp
            outputs = tracker.step_centertrack(inputs, time_lag)

            # # Debug
            # cv2.imshow('key_ctl', np.empty((200, 200), dtype=np.uint8))
            # key = cv2.waitKey(0)
            # if key == 27: exit(0)
            # print(f"Frame: {local_frame_num}, key_token: {sample_token}")
            # for sensor_name in RADAR_NAMES:
            #     if radar_sample_data[sensor_name] is not None:
            #         sample_data = radar_sample_data[sensor_name]
            #         print(f"{sensor_name:20s}: {sample_data['timestamp'] * 1e-6:.2f}-{sample_data['sample_token']}")
            # print()

            # Save tracking result to dictionary (use first `not None sample data token` to record)
            for sensor_name in RADAR_NAMES:
                if radar_sample_data[sensor_name] is not None:
                    points_trk = []
                    for item in outputs:
                        if 'active' in item and item['active'] < args.min_hits:
                            continue
                        nusc_trk = {
                            'sample_token': sample_token,
                            'token': radar_sample_data[sensor_name]['token'], 
                            'next': radar_sample_data[sensor_name]['next'], 
                            'ego_pose_token': ego_pose_token, 
                            'is_key_frame': radar_sample_data[sensor_name]['is_key_frame'], 
                            'translation': item['pose'],
                            'velocity': item['velocity'],
                            'tracking_id': str(item['tracking_id']),
                        }
                        points_trk.append(nusc_trk)
                    # use `not None sample data token` to record all results
                    radar_points_trk["results"].update({radar_sample_data[sensor_name]['token']: deepcopy(points_trk)})
                    # use `sample token` to record key frames resutls
                    if radar_sample_data[sensor_name]['is_key_frame']:
                        radar_points_trk["key_results"].update({sample_token: deepcopy(points_trk)})
                        key_frame_num += 1
                    break

            # Visualize the tracking point cloud in world frame
            if viz:
                positions = [[], []]
                tracks = [[], []]
                track_color = []
                for point in radar_pointcloud:
                    positions[0].append(point[0])
                    positions[1].append(point[1])
                for point in points_trk:
                    tracks[0].append(point['translation'][0])
                    tracks[1].append(point['translation'][1])
                    track_color.append(int(point['tracking_id']) % 255)
                radar_trackers = radar_traj.save_path(points_trk)
                plt.clf()
                plt.xlabel('x axis')
                plt.ylabel('y axis')
                ego_pose = nusc.get('ego_pose', ego_pose_token)['translation']
                plt.xlim((ego_pose[0] - 100, ego_pose[0] + 100))
                plt.ylim((ego_pose[1] - 100, ego_pose[1] + 100))
                plt.scatter(positions[0], positions[1], label='radar pointcloud', c='r')
                plt.scatter(tracks[0], tracks[1], label='tracks', c=track_color, cmap='magma')
                plt.scatter(ego_pose[0], ego_pose[1], label='ego pose')
                # Draw trajectory
                for point in points_trk:
                    id = point['tracking_id']
                    if len(radar_trackers[id].locations) <= 1:
                        continue
                    line = [[], []]
                    for p in radar_trackers[id].locations:
                        line[0].append(p[0])
                        line[1].append(p[1])
                    plt.plot(line[0], line[1], label='trajectory', c='k')
                plt.legend()
                plt.tight_layout()
                plt.pause(0.001)
                fig.canvas.mpl_connect('key_press_event', on_key)

            # Get next radar sample data if exist
            for sensor_name in RADAR_NAMES:
                try:
                    sample_data = nusc.get('sample_data', radar_sample_data[sensor_name]['next'])
                    sample_token = sample_data['sample_token']
                    radar_sample_data.update({sensor_name: sample_data})
                except:
                    radar_sample_data.update({sensor_name: None})

            # Synchronize
            key_exist = False
            for sensor_name in RADAR_NAMES:
                sample_data = radar_sample_data[sensor_name]
                if sample_data is None:
                    continue
                if sample_data['is_key_frame']:
                    key_exist = True
                    key_token = sample_data['sample_token']
                    break
            for sensor_name in RADAR_NAMES:
                sample_data = radar_sample_data[sensor_name]
                if sample_data is None:
                    continue
                while key_exist and not sample_data['is_key_frame']:
                    try:
                        sample_data = nusc.get('sample_data', sample_data['next'])
                        radar_sample_data.update({sensor_name: sample_data})
                    except:
                        break
            if key_exist:
                sample_token = key_token
            else:
                for sensor_name in RADAR_NAMES:
                    sample_data = radar_sample_data[sensor_name]
                    if sample_data is None:
                        continue
                    sample_token = sample_data['sample_token']
                    break
            # timestamps = np.zeros(5, dtype=np.float)
            # i = 0
            # for sensor_name in RADAR_NAMES:
            #     i += 1
            #     sample_data = radar_sample_data[sensor_name]
            #     if sample_data is None:
            #         continue
            #     timestamp = sample_data['timestamp']
            #     print(f"{sensor_name:20s}: {timestamp * 1e-6}")
            #     timestamps[i-1] = timestamp
            # diff = (timestamps - timestamps[0]) * 1e-6
            # print(f"diff: {diff}")

            # Finish or not (in one scene)
            live_sensor_num = 0
            for sensor_name in RADAR_NAMES:
                if radar_sample_data[sensor_name] is not None:
                    live_sensor_num += 1
            if live_sensor_num == 0:
                break

        total_frame_num += local_frame_num

    # calculate computation time
    end = time.time()
    second = (end - start)
    speed = total_frame_num / second
    print("The speed is {} FPS".format(speed))
    print(f"total frame number: {total_frame_num}")
    print(f"key frame number: {key_frame_num}")

    # add meta info to writen result file
    radar_points_trk["meta"] = {
        "use_camera": False,
        "use_lidar": False,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }

    # write result file
    run_time = get_current_datetime()
    output_file = os.path.join(args.out_dir, run_time)
    mkdir_or_exist(output_file)
    with open(os.path.join(output_file, 'radar_tracking_result_13Hz.json'), "w") as f:
        json.dump(radar_points_trk, f)
    print(f"Radar Tracking results (13Hz) are saved to {output_file}/radar_tracking_result_13Hz.json\n")
    return speed

''' ======================================================================================================== '''
''' ======================================================================================================== '''
''' ==================================== For making radar PC with 13 Hz ==================================== '''
''' ======================================================================================================== '''
''' ======================================================================================================== '''

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

def get_radar_PC(args, freq=13, viz=False):
    
    if viz:
        fig = plt.figure(figsize=(12, 8))
        def on_key(event):
            if event.key == 'escape':
                sys.exit(f"Exit by pressing {event.key}")

    # v1.0-trainval (default: validation set)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    scenes_names = splits.val

    numOfScene = len(nusc.scene)
    total_frame_num = 0
    radar_PCs = {}
    for scene_num in tqdm(range(numOfScene)):
        # Get first scene token
        my_scene = nusc.scene[scene_num]
        scene_name = my_scene['name']
        if scene_name not in scenes_names:
            continue

        sample_token = my_scene['first_sample_token']
        current_scene_name = scene_name
        sample = nusc.get('sample', sample_token)
        radar_sample_data = {
            'RADAR_FRONT': None, 
            'RADAR_FRONT_LEFT': None, 
            'RADAR_FRONT_RIGHT': None, 
            'RADAR_BACK_LEFT': None, 
            'RADAR_BACK_RIGHT': None,
        }
        for sensor_name in RADAR_NAMES:
            sample_data = nusc.get('sample_data', sample['data'][sensor_name])
            radar_sample_data.update({sensor_name: sample_data})

        local_frame_num = 0
        while scene_name == current_scene_name:
            sample = nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            current_scene_name = nusc.get('scene', scene_token)['name']
            local_frame_num += 1

            # # Debug
            # cv2.imshow('key_ctl', np.empty((200, 200), dtype=np.uint8))
            # key = cv2.waitKey(0)
            # if key == 27: exit(0)
            # print(f"Frame: {local_frame_num}, key_token: {sample_token}")
            # for sensor_name in RADAR_NAMES:
            #     if radar_sample_data[sensor_name] is not None:
            #         sample_data = radar_sample_data[sensor_name]
            #         print(f"{sensor_name:20s}: {sample_data['timestamp'] * 1e-6:.2f}-{sample_data['sample_token']}-Key:{sample_data['is_key_frame']}")
            # print()

            # Get radar pointcloud
            point_cloud = []
            for sensor_name in RADAR_NAMES:
                if radar_sample_data[sensor_name] is None:
                    continue
                ego_pose_token = radar_sample_data[sensor_name]['ego_pose_token']
                calib_token = radar_sample_data[sensor_name]['calibrated_sensor_token']
                points = RadarPointCloud.from_file(os.path.join(args.dataroot, radar_sample_data[sensor_name]['filename'])).points.T
                points = points[:, [0, 1, 2, 8, 9]] # x, y, z, vx_comp, vy_comp
                points = pc2world(nusc, points, ego_pose_token, calib_token, sensor_name, inverse=False)
                point_cloud.append(points)
            point_cloud = np.concatenate(point_cloud, axis=0).tolist()

            # Visualize the point cloud in world frame
            if viz:
                positions = [[], []]
                for point in point_cloud:
                    positions[0].append(point[0])
                    positions[1].append(point[1])
                plt.clf()
                plt.xlabel('x axis')
                plt.ylabel('y axis')
                ego_pose = nusc.get('ego_pose', ego_pose_token)['translation']
                plt.xlim((ego_pose[0] - 100, ego_pose[0] + 100))
                plt.ylim((ego_pose[1] - 100, ego_pose[1] + 100))
                plt.scatter(positions[0], positions[1], label='radar pointcloud', c='r')
                plt.scatter(ego_pose[0], ego_pose[1], label='ego pose', c='b')
                plt.legend()
                plt.tight_layout()
                plt.pause(0.001)
                fig.canvas.mpl_connect('key_press_event', on_key)

            # Save to dictionary (use first `not None sample data token` to record)
            for sensor_name in RADAR_NAMES:
                if radar_sample_data[sensor_name] is not None:
                    radar_PC = {
                        'sample_token': sample_token, 
                        'token': radar_sample_data[sensor_name]['token'], 
                        'next': radar_sample_data[sensor_name]['next'], 
                        'ego_pose_token': ego_pose_token,
                        'is_key_frame': radar_sample_data[sensor_name]['is_key_frame'], 
                        'points': point_cloud, 
                    }
                    radar_PCs.update({radar_sample_data[sensor_name]['token']: radar_PC})
                    break

            # Get next radar sample data if exist
            for sensor_name in RADAR_NAMES:
                try:
                    sample_data = nusc.get('sample_data', radar_sample_data[sensor_name]['next'])
                    # sample_token = sample_data['sample_token']
                    radar_sample_data.update({sensor_name: sample_data})
                except:
                    radar_sample_data.update({sensor_name: None})

            # Synchronize
            key_exist = False
            for sensor_name in RADAR_NAMES:
                sample_data = radar_sample_data[sensor_name]
                if sample_data is None:
                    continue
                if sample_data['is_key_frame']:
                    key_exist = True
                    key_token = sample_data['sample_token']
                    break
            for sensor_name in RADAR_NAMES:
                sample_data = radar_sample_data[sensor_name]
                if sample_data is None:
                    continue
                while key_exist and not sample_data['is_key_frame']:
                    try:
                        sample_data = nusc.get('sample_data', sample_data['next'])
                        radar_sample_data.update({sensor_name: sample_data})
                    except:
                        break
            if key_exist:
                sample_token = key_token
            else:
                for sensor_name in RADAR_NAMES:
                    sample_data = radar_sample_data[sensor_name]
                    if sample_data is None:
                        continue
                    sample_token = sample_data['sample_token']
                    break
            # timestamps = np.zeros(5, dtype=np.float)
            # i = 0
            # for sensor_name in RADAR_NAMES:
            #     i += 1
            #     sample_data = radar_sample_data[sensor_name]
            #     if sample_data is None:
            #         continue
            #     timestamp = sample_data['timestamp']
            #     print(f"{sensor_name:20s}: {timestamp * 1e-6}")
            #     timestamps[i-1] = timestamp
            # diff = (timestamps - timestamps[0]) * 1e-6
            # print(f"diff: {diff}")

            # Finish or not (in one scene)
            live_sensor_num = 0
            for sensor_name in RADAR_NAMES:
                if radar_sample_data[sensor_name] is not None:
                    live_sensor_num += 1
            if live_sensor_num == 0:
                break

        total_frame_num += local_frame_num

    print(f"Total frame numbers: {total_frame_num}")
            
    # Add radar PC to json file
    meta = {
        'total_frame_nbr': total_frame_num,
        'frequency': freq, 
    }
    run_time = get_current_datetime()
    output_file = os.path.join(args.out_dir, run_time)
    mkdir_or_exist(output_file)
    with open(os.path.join(output_file, 'radar_PC_13Hz_with_vcomp.json'), "w") as f:
        json.dump({
            'meta': meta,
            'radar_PCs': radar_PCs, 
        }, f)
        print(f"radar_PC_13Hz_with_vcomp.json is saved to {output_file}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--dataroot", type=str, default="/data/nuscenes", help='Nuscenes dataset directory')
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--out_dir", type=str, default='/data/radar_PC')
    parser.add_argument("--tracker", type=str, default='KF', help='use KF or PointTracker')
    parser.add_argument("--hungarian", type=bool, default=False, help='use hungarian or greedy')
    parser.add_argument("--use_vel", type=bool, default=False, help='use radar velocity or not')
    parser.add_argument("--max_age", type=int, default=5)
    parser.add_argument("--min_hits", type=int, default=2)
    parser.add_argument("--viz_trk", action='store_true')
    parser.add_argument("--viz_pc", action='store_true')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # get_radar_PC(args, freq=13, viz=args.viz_pc)
    main(args, viz=args.viz_trk)