import argparse
import copy
import os
import sys
import time
import json
import math

import numpy as np
import torch

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy

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
    ):
        # self.nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
        self.det_res = self.load_detections(result_path)
        self.frames = self.load_frames_meta(frame_meta_path)
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

    def get_det_meta(self):
        return self.det_res['meta']

    def get_det_results(self, bbox_th):
        return self.filter_box(self.det_res['results'], bbox_th)

    def get_frames_meta(self):
        return self.frames['frames']

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

# Tracking usage ==========================================

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="data/track_results")

    parser.add_argument("--min_hits", type=int, default=1)
    parser.add_argument("--det_th", type=float, default=0.1)
    # parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--dataroot", type=str, default='data/nuscenes')
    parser.add_argument("--detection_path", type=str, default='data/detection_result.json')
    parser.add_argument("--frames_meta_path", type=str, default='data/frames_meta.json')
    args, opts = parser.parse_known_args()

    # Build data preprocessing for track
    dataset = nusc_dataset(
        nusc_version='v1.0-trainval',
        split=args.split,
        nusc_path=args.dataroot,
        result_path=args.detection_path,
        frame_meta_path=args.frames_meta_path,
    )

    from tracker import PubTracker
    tracker = PubTracker(
        hungarian=False,
        max_age=6,
        active_th=1,
        min_hits=args.min_hits,
        score_update=None,
        deletion_th=0,
        detection_th=args.det_th,
        dataset='Nuscenes',
    )
    # prepare writen output file
    nusc_annos_trk = {
        "results": {},
        "meta": None,
    }
    
    # Load detection results
    detections = dataset.get_det_results(args.bbox_score)
    frames = dataset.get_frames_meta()
    len_frames = len(frames)

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
            last_time_stamp = timestamp

        # calculate time between two frames
        time_lag = (timestamp - last_time_stamp)
        last_time_stamp = timestamp

        # get tracklets of current frame
        det = detections[token]
        outputs = tracker.step_centertrack(det, time_lag)

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

    # calculate computation time
    end = time.time()
    second = (end - start)
    speed = len_frames / second
    print("======")
    print("The speed is {} FPS".format(speed))
    print("tracking results have {} frames".format(len(nusc_annos_trk["results"])))

    nusc_annos_trk["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    # write result file
    root_path = args.out_dir
    track_res_path = os.path.join(root_path, 'tracking_result.json')
    mkdir_or_exist(os.path.dirname(track_res_path))
    with open(track_res_path, "w") as f:
        json.dump(nusc_annos_trk, f)
    print(f"tracking results write to {track_res_path}\n")

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
