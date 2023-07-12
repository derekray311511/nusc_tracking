import argparse
import copy
import os
import cv2
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
from utils.utils import log_parser_args, mkdir_or_exist, cal_func_time
from utils.box_utils import get_3d_box_8corner, get_3d_box_2corner
from utils.box_utils import nms, is_points_inside_obb
from utils.geometry_utils import *
from utils.visualizer import TrackVisualizer
from tracker import PubTracker as lidarTracker

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

class nusc_dataset:
    def __init__(
        self, 
        nusc_version='v1.0-trainval', 
        split='val', 
        nusc_path='/data/nuscenes',
        det_result_path='data/detection_result.json',
        frame_meta_path='data/frames_meta.json',
        radar_pc_path='data/radar_PC/radar_PC_13Hz_with_vcomp.json',
    ):
        # self.nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
        self.det_res = self.load_detections(det_result_path)
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

class RadarSegmentor:
    """
    Segment radar targets using LiDAR detection

    param:

    """
    def __init__(self):
        self.id_count = 0
        self.radarSegmentation = None   # radar segmentation that is already grouped by LiDAR

    def reset(self):
        self.id_count = 0
        self.radarSegmentation = None

    def run(self, radarTargets, lidarDet):
        """ (Global pose)
        Returns radar targets with id: list of [x, y, z, vx, vy, id]

        radarTargets: list of [x, y, z, vx, vy]
        lidarDet: nusc_det : list of {'translation': [x, y, z], 'rotation': [w, x, y, z], 'size': [x, y, z], 'velocity': [vx, vy], 'detection_name': s, 'detection_score': s, 'sample_token': t}
        """
        radarSegmentation = []

        # Sort the bboxes by detection scores
        lidarDet.sort(reverse=True, key=lambda box:box['detection_score'])

        # Prepare radar center x, y
        pts = []
        for p in radarTargets:
            pts.append(p[:2])
        radarTargets = np.array(radarTargets)
        pts = np.array(pts)

        # Segment radar targets
        for det in lidarDet:
            center = det['translation'][:2]
            size = [det['size'][1], det['size'][0]]
            row, pitch, angle = euler_from_quaternion(q_to_xyzw(det['rotation']))
            inbox_idxs = is_points_inside_obb(pts, center, size, angle)
            
            # Give radar targets segmentation id
            temp_targets = radarTargets[inbox_idxs]
            if len(temp_targets) > 0:
                seg = np.hstack([temp_targets, (np.ones(len(temp_targets)) * (self.id_count)).reshape(-1, 1)])
                radarSegmentation.append(seg)
                self.id_count += 1
                radarTargets = radarTargets[~inbox_idxs]
                pts = pts[~inbox_idxs]

        # Give none segmented targets -1 as id
        radarTargets = np.hstack([radarTargets, -np.ones(len(radarTargets)).reshape(-1, 1)])

        if len(radarSegmentation) > 0:
            radarSegmentation = np.vstack([np.array(seg) for seg in radarSegmentation])
            self.radarSegmentation = np.concatenate((radarSegmentation, radarTargets), axis=0)
        else:
            self.radarSegmentation = radarTargets

        return self.radarSegmentation

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LRFusion Tracking")
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
    parser.add_argument("--vel_fusion", type=int, default=0)
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--dataroot", type=str, default='data/nuscenes')
    parser.add_argument("--workspace", type=str, default='/home/Student/Tracking')
    parser.add_argument("--detection_path", type=str, default='data/detection_result.json')
    parser.add_argument("--frames_meta_path", type=str, default='data/frames_meta.json')
    parser.add_argument("--radar_fusion", type=int, default=1)
    parser.add_argument("--radar_pc_path", type=str, default='data/radar_PC/radar_PC_13Hz_with_vcomp.json')

    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--viz", action="store_true")
    return parser

def main(parser) -> None:
    args, opts = parser.parse_known_args()

    # Notice
    if args.radar_fusion:
        print(f"======\nRunning LRFusion using LiDAR detection and Radar targets\n======")
    if args.vel_fusion:
        print("Using LiDAR detection to refine radar velocity")

    # Prepare results saving path / Copy parameters and code
    root_path = args.out_dir
    track_res_path = os.path.join(root_path, 'tracking_result.json')
    mkdir_or_exist(os.path.dirname(track_res_path))
    if args.save_log:
        log_parser_args(root_path, args)
        shutil.copyfile(os.path.join(args.workspace, 'tools/early_fusion.sh'), os.path.join(root_path, 'early_fusion.sh'), follow_symlinks=True)
        shutil.copyfile(os.path.join(args.workspace, 'tools/early_fusion.py'), os.path.join(root_path, 'early_fusion.py'), follow_symlinks=True)

    # Build data preprocessing for track
    dataset = nusc_dataset(
        nusc_version='v1.0-trainval',
        split=args.split,
        nusc_path=args.dataroot,
        det_result_path=args.detection_path,
        frame_meta_path=args.frames_meta_path,
        radar_pc_path=args.radar_pc_path,
    )

    # Build Trackers and Segmentor
    radarSeg = RadarSegmentor()
    lidar_tracker = lidarTracker(
        hungarian=False,
        max_age=args.max_age,
        active_th=args.active_th,
        min_hits=args.min_hits,
        score_update=args.update_function,
        noise=args.score_decay,
        deletion_th=args.del_th,
        detection_th=args.det_th,
        use_vel=args.use_vel,
        tracker=args.tracker,
    )

    # Build Vizualizer
    if args.viz:
        trackViz = TrackVisualizer(
            nusc_trk_name=NUSCENES_TRACKING_NAMES, 
            range=(200, 200), 
            windowSize=(1000, 1000), 
            imgSize=(1600, 1600),
            duration=0.1
        )

    # prepare writen output file
    nusc_LiDAR_trk = {
        "results": {},
        "meta": None,
    }
    nusc_Radar_trk = {
        "results": {},
        "meta": None,
    }
    nusc_trk = {
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
        ego_pose = frames[i]['ego_pose']
        name = "{}-{}".format(timestamp, token)

        # Reset tracker if this is first frame of the sequence
        if frames[i]['first']:
            lidar_tracker.reset()
            # radar_tracker.reset()
            # fusion_module.reset()
            last_time_stamp = timestamp

        # calculate time between two frames
        time_lag = (timestamp - last_time_stamp)
        last_time_stamp = timestamp

        # Get LiDAR detection for current frame
        det = detections[token]

        if args.use_nms:
            det, _ = nms(det, args.nms_th)

        # Get Radar targets
        radar_pc = dataset.get_key_radar_pc(token)
        # Transform list of [x, y, z, vx, vy] to list of {'pose':[x, y], 'vel':mirror_velocity, 'vel_comp':[vx, vy]}
        radar_targets = []
        for point in radar_pc:
            radar_targets.append({
                'pose': point[:2],
                'translation': point[:2],
                'vel': np.sqrt(point[3]**2 + point[4]**2),
                'vel_comp': point[3:5],
            })

        # Radar segmentation with LiDAR detection
        det_copy = deepcopy(det)
        segResult = radarSeg.run(radar_pc, det)
        print(f"Radar segmented to {radarSeg.id_count} groups")
        radarSeg.reset()

        # Tracking for LiDAR and Radar respectively
        lidar_trks = lidar_tracker.step_centertrack(det, time_lag)
        lidar_active_trks = []
        for trk in lidar_trks:
            if not trk['active']: continue
            trk['size'][0], trk['size'][1] = 1.2 * trk['size'][0], 1.2 * trk['size'][1]
            lidar_active_trks.append(trk)

        # Fusion module (ID arrangement)


        # Vizsualize (realtime)
        if args.viz:

            if trackViz.play:
                key = cv2.waitKey(int(trackViz.duration * 1000))
            else:
                key = cv2.waitKey(0)

            if key == 27: # esc
                cv2.destroyAllWindows()
                exit(0)
            elif key == 32: # space
                trackViz.play = not trackViz.play
            elif key == 43: # +
                trackViz.duration *= 2
                print(f"Viz duration set to {trackViz.duration}")
            elif key == 45: # -
                trackViz.duration *= 0.5
                print(f"Viz duration set to {trackViz.duration}")

            trans = dataset.get_4f_transform(ego_pose, inverse=True)
            trackViz.draw_ego_car(img_src="/data/car1.png")
            _, delay1 = cal_func_time(trackViz.draw_radar_seg, radarSeg=segResult, trans=trans, colorID=True, contours=False)
            _, delay2 = cal_func_time(trackViz.draw_det_bboxes, nusc_det=det_copy, trans=trans)
            _, delay3 = cal_func_time(trackViz.draw_det_bboxes, nusc_det=lidar_active_trks, trans=trans, BGRcolor=(52, 171, 235))
            trackViz.show()
            print(f"viz pts delay: {delay1}")
            print(f"viz trk delay: {delay3}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(get_parser())