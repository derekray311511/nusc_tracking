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
import yaml

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy
from utils.utils import log_parser_args, mkdir_or_exist, cal_func_time, get_current_datetime
from utils.utils import encodeCategory, decodeCategory
# from utils.box_utils import get_3d_box_8corner, get_3d_box_2corner
from utils.box_utils import nms, is_points_inside_obb
from utils.geometry_utils import *
from utils.visualizer import TrackVisualizer
from utils.nusc_eval import nusc_eval
from collections import deque
from geometry import *
from pre_processing import *
from tracker import PubTracker as lidarTracker
from early_fusion_tracker import PubTracker as radarTracker
from early_fusion_fusion import Fusion

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]
RADAR_TRACKING_NAMES = NUSCENES_TRACKING_NAMES + ['background']

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

class TrackHistory:
    def __init__(self, max_frames=5):
        self.history = deque(maxlen=max_frames)

    def __len__(self):
        return len(self.history)

    def update(self, new_frame):
        self.history.append(new_frame)

    def get_history(self):
        return list(self.history)

class nusc_dataset:
    def __init__(self, cfg):
        self.nusc_version = cfg["nusc_version"]
        self.dataroot = cfg["dataroot"]
        # self.nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=True)
        self.gt = self.load_groundTruth(cfg["groundTruth_path"])
        self.split = cfg["split"]
        self.categories = NUSCENES_TRACKING_NAMES
        self.det_res = self.load_detections(cfg["lidar_det_path"])
        self.frames = self.load_frames_meta(cfg["frames_meta_path"])
        self.radar_PCs = self.load_radar_pc(cfg["radar_pc_path"])
        self.key_radar_PCs = self.load_key_radar_pc()
        if self.split == 'train':
            self.scene_names = splits.train
        elif self.split == 'val':
            self.scene_names = splits.val
        elif self.split == 'test':
            sys.exit("Not support test data yet!")
        else:
            sys.exit(f"No split type {self.split}!")

    def load_groundTruth(self, path):
        with open(path, 'rb') as f:
            gt = json.load(f)['samples']
        return gt

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

    def get_det_results(self, bbox_th=0.0, categories=None):
        """
        bbox_th: Filter bboxes which have scores below bbox_th
        categories: Filter bboxes by category (if categories is None then doesn't filter by category) 
        """
        return self.filter_box(self.det_res['results'], bbox_th, categories)

    def get_frames_meta(self):
        return self.frames['frames']

    def get_groundTruth(self):
        return self.gt
    
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

    def filter_box(self, det_res, th=0.0, categories=None):
        print("======")
        print(f"Filtering bboxes by threshold {th} and category...", end='')
        ret_dict = {}
        for i, (token, bboxes) in enumerate(det_res.items()):
            ret_bboxes = []
            for bbox in bboxes:
                if bbox['detection_score'] < th:
                    continue
                if categories and bbox['detection_name'] not in categories:
                    continue
                ret_bboxes.append(bbox)
            ret_dict.update({token: ret_bboxes})
        print("Done.")
        return ret_dict

class RadarSegmentor:
    """
    Segment radar targets using LiDAR detection

    param:
        categories: list of nuscenes detection category to be used
    """
    def __init__(self, cfg):
        self.categories = cfg["detCategories"]
        self.track_cat = NUSCENES_TRACKING_NAMES
        self.expand_rotio = cfg["expand_ratio"]
        self.fuse_vel = cfg["fuse_velocity"]
        self.id_count = 0
        self.radarSegmentation = None   # radar segmentation that is already grouped by LiDAR

    def reset(self):
        self.id_count = 0
        self.radarSegmentation = None

    def _filterDet(self, det, thr=0.0):
        filtered_det = []
        for obj in det:
            if obj['detection_name'] in self.categories and obj['detection_score'] >= thr:
                filtered_det.append(obj)
        return filtered_det

    def _filterDetByDist(self, det, ego_pose, dist_th=0.0):
        # dist_th == 0 means no filtering
        if dist_th == 0 or ego_pose is None:
            return det

        filtered_det = []
        trans = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)
        inv_trans = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        det = det_transform(det, trans) # world to local coordinates
        for obj in det:
            # if np.linalg.norm(obj['translation'][:2]) <= dist_th: # circle range
            #     filtered_det.append(obj)
            if abs(obj['translation'][0]) <= dist_th and abs(obj['translation'][1]) <= dist_th: # square range
                filtered_det.append(obj)
        filtered_det = det_transform(filtered_det, inv_trans) # local back to world coordinates

        return filtered_det

    def run(self, radarTargets, lidarDet, score_thr=0.0, dist_th=0.0, ego_pose=None):
        """ (Global pose)
        Returns radar targets with id: list of [x, y, z, vx, vy, id, cat_num, score]

        radarTargets: list of [x, y, z, vx, vy]
        lidarDet: nusc_det : list of {'translation': [x, y, z], 'rotation': [w, x, y, z], 'size': [x, y, z], 'velocity': [vx, vy], 'detection_name': s, 'detection_score': s, 'sample_token': t}
        """
        radarSegmentation = []

        # Filter LiDAR detection by name, score and distance
        lidarDet = deepcopy(lidarDet)
        lidarDet = self._filterDet(lidarDet, score_thr)
        lidarDet = self._filterDetByDist(lidarDet, ego_pose, dist_th)

        # Sort the bboxes by detection scores
        lidarDet.sort(reverse=True, key=lambda box:box['detection_score'])

        # Prepare radar center x, y
        pts = np.array([p[:2] for p in radarTargets])  # Extract x, y coordinates
        radarTargets = np.array(radarTargets)

        # Create a mask for filtering points and targets
        remaining_idxs = np.arange(len(pts))

        # Set init id and cat
        radarTargets = np.hstack([radarTargets, -np.ones((len(radarTargets), 2))])
        radarTargets = np.hstack([radarTargets, 0.5*np.ones((len(radarTargets), 1))])

        # Segment radar targets
        for det in lidarDet:
            ratio = self.expand_rotio
            center = det['translation'][:2]
            size = [det['size'][1] * ratio[1] , det['size'][0] * ratio[0]]
            det_vel = det['velocity']
            cat_num = encodeCategory([det['detection_name']], self.track_cat)[0]
            row, pitch, angle = euler_from_quaternion(q_to_xyzw(det['rotation']))
            inbox_idxs = is_points_inside_obb(pts[remaining_idxs], center, size, angle)
            
            # Use boolean masks for filtering instead of indexing
            temp_idxs = remaining_idxs[inbox_idxs]
            if len(temp_idxs) == 0:
                continue

            # Give radar targets segmentation id
            radarTargets[temp_idxs, 5] = self.id_count

            # Give radar targets category name
            radarTargets[temp_idxs, 6] = cat_num

            # Give radar targets detection score
            radarTargets[temp_idxs, 7] = det['detection_score']

            # Use LiDAR detection velocity
            if self.fuse_vel:
                for idx in temp_idxs:
                    radarTargets[idx, 3:5] = det_vel[:2]

            radarSegmentation.append(radarTargets[temp_idxs])
            self.id_count += 1
            remaining_idxs = remaining_idxs[~inbox_idxs]  # Update remaining_idxs using the opposite mask

        # Mark remaining points as none segmented targets
        # Give none segmented targets -1 as id
        remaining_targets = radarTargets[remaining_idxs]
        self.radarSegmentation = radarTargets

        return radarTargets

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LRFusion Tracking")
    parser.add_argument("config", metavar="CONFIG_FILE")
    parser.add_argument("--workspace", type=str, default="/home/Student/Tracking")
    parser.add_argument("--out-dir", type=str, default="/data/early_fusion_track_results/Test")
    parser.add_argument("--out_time", action="store_true")
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--show_delay", action="store_true")
    return parser

def main(parser) -> None:
    args, opts = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    # set output dir
    if args.out_time:
        temp = args.out_dir.split('/')
        out_dirname = get_current_datetime() + "_" + temp[-1]
        args.out_dir = "/" + os.path.join(*temp[:-1], out_dirname)
    root_path = args.out_dir
    
    cfg["EVALUATE"]["out_dir"] = os.path.join(root_path, 'eval')
    cfg["EVALUATE"]["out_dir"] += "_" + str(cfg["EVALUATE"]["custom_range"]["car"]) + "_" + str(cfg["EVALUATE"]["dist_th_tp"])

    # Prepare results saving path / Copy parameters and code
    print(f"CONFIG:\n{yaml.dump(cfg, sort_keys=False)}\n")
    track_res_path = os.path.join(root_path, 'tracking_result.json')
    mkdir_or_exist(os.path.dirname(track_res_path))
    if args.save_log:
        log_parser_args(root_path, args)
        shutil.copyfile(os.path.join(args.workspace, 'tools/early_fusion.sh'), os.path.join(root_path, 'early_fusion.sh'), follow_symlinks=True)
        shutil.copyfile(os.path.join(args.workspace, 'tools/early_fusion.py'), os.path.join(root_path, 'early_fusion.py'), follow_symlinks=True)
        shutil.copyfile(os.path.join(args.workspace, 'tools/early_fusion_fusion.py'), os.path.join(root_path, 'early_fusion_fusion.py'), follow_symlinks=True)
        shutil.copyfile(os.path.join(args.workspace, 'tools/early_fusion_tracker.py'), os.path.join(root_path, 'early_fusion_tracker.py'), follow_symlinks=True)
        shutil.copyfile(os.path.join(args.workspace, 'configs/early_fusion.yaml'), os.path.join(root_path, 'early_fusion.yaml'), follow_symlinks=True)

    # Build data preprocessing for track
    dataset = nusc_dataset(cfg["DATASET"])

    # Build Trackers and Segmentor
    expand_ratio = cfg["SEGMENTOR"]["expand_ratio"]
    radarSeg = RadarSegmentor(cfg["SEGMENTOR"])
    lidar_tracker = lidarTracker(
        hungarian=cfg["LIDAR_TRACKER"]["hungarian"],
        max_age=cfg["LIDAR_TRACKER"]["max_age"],
        active_th=cfg["LIDAR_TRACKER"]["active_th"],
        min_hits=cfg["LIDAR_TRACKER"]["min_hits"],
        score_update=cfg["LIDAR_TRACKER"]["update_function"],
        noise=cfg["LIDAR_TRACKER"]["score_decay"],
        deletion_th=cfg["LIDAR_TRACKER"]["deletion_th"],
        detection_th=cfg["LIDAR_TRACKER"]["detection_th"],
        use_vel=cfg["LIDAR_TRACKER"]["use_vel"],
        tracker=cfg["LIDAR_TRACKER"]["tracker"],
    )
    radar_tracker = radarTracker(
        hungarian=cfg["RADAR_TRACKER"]["hungarian"],
        max_age=cfg["RADAR_TRACKER"]["max_age"],
        active_th=cfg["RADAR_TRACKER"]["active_th"],
        min_hits=cfg["RADAR_TRACKER"]["min_hits"],
        score_update=cfg["RADAR_TRACKER"]["update_function"],
        noise=cfg["RADAR_TRACKER"]["score_decay"],
        deletion_th=cfg["RADAR_TRACKER"]["deletion_th"],
        detection_th=cfg["RADAR_TRACKER"]["detection_th"],
        use_vel=cfg["RADAR_TRACKER"]["use_vel"],
        tracker=cfg["RADAR_TRACKER"]["tracker"],
    )
    fusion_module = Fusion(
        hungarian=cfg["FUSION"]["hungarian"],
        decay1=cfg["FUSION"]["decay1"],
        decay2=cfg["FUSION"]["decay2"],
        star=cfg["FUSION"]["star"],
        del_th=cfg["FUSION"]["del_th"],
        v_min=cfg["FUSION"]["v_min"],
        v_max=cfg["FUSION"]["v_max"],
        v_weight=cfg["FUSION"]["v_weight"],
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
    nusc_fusion_trk = {
        "results": {},
        "meta": None,
    }
    nusc_seg_result = {
        "results": {},
        "meta": None,
    }

    # Load detection results
    detections = dataset.get_det_results(cfg["DETECTION"]["bbox_score"], NUSCENES_TRACKING_NAMES)
    frames = dataset.get_frames_meta()
    gts = dataset.get_groundTruth()
    len_frames = len(frames)

    # Build Vizualizer
    if args.viz:
        trackViz = TrackVisualizer(
            windowName='Intermediate',
            **cfg["VISUALIZER"],
        )
        trackViz2 = TrackVisualizer(
            windowName='Fusion result',
            **cfg["VISUALIZER"],
        )
        # trackViz3 = TrackVisualizer(
        #     windowName='Trajectory',
        #     **cfg["VISUALIZER"],
        # )
        winList = [trackViz, trackViz2]
        for win in winList:
            cv2.createTrackbar('Frame', win.windowName, 0, len_frames, lambda x: None)
            cv2.setTrackbarPos('Frame', win.windowName, 0)
        
        # cfg["VISUALIZER"]["trkHist"] = cfg["VISUALIZER"]["fusionBox"]
        # cfg["VISUALIZER"]["trkHist"]["colorName"] = False
        # cfg["VISUALIZER"]["trkHist"]["colorID"] = True
        # cfg["VISUALIZER"]["trkHist"]["draw_vel"] = False
        # cfg["VISUALIZER"]["trkHist"]["legend"] = False

    # start tracking *****************************************
    print("Begin Tracking\n")
    start = time.time()
    thrFrames = []
    pbar = tqdm(total=len_frames)
    # for i in tqdm(range(len_frames)):
    i = -1
    while i < len_frames:

        if args.viz:
            if trackViz.play:
                i += 1
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
            elif key == ord('i'):
                cfg["VISUALIZER"]["trkBox"]["draw_id"] = not cfg["VISUALIZER"]["trkBox"]["draw_id"]
                cfg["VISUALIZER"]["radarTrkBox"]["draw_id"] = not cfg["VISUALIZER"]["radarTrkBox"]["draw_id"]
                cfg["VISUALIZER"]["fusionBox"]["draw_id"] = not cfg["VISUALIZER"]["fusionBox"]["draw_id"]
                cfg["VISUALIZER"]["trkHist"]["draw_id"] = not cfg["VISUALIZER"]["trkHist"]["draw_id"]
            elif key == ord('g'):
                for win in winList:
                    win.grid = not win.grid
            elif key == ord('d'):
                i += 1
        else:
            i += 1

        thrFrames.append(i)

        if i >= len_frames:
            break

        if args.viz:
            for win in winList:
                cv2.setTrackbarPos('Frame', win.windowName, i)

        # get frameID (=token)
        token = frames[i]['token']
        timestamp = frames[i]['timestamp']
        ego_pose = frames[i]['ego_pose']
        gt = gts[i]['anns']
        for obj in gt:
            # obj['velocity'] = np.array([0.0, 0.0])
            obj['detection_score'] = 1.0
        name = "{}-{}".format(timestamp, token)

        # Reset tracker if this is first frame of the sequence
        if frames[i]['first']:
            lidar_tracker.reset()
            radar_tracker.reset()
            # fusion_module.update_scene_velos(detections, frames, i)  # get median velocities of current scene
            fusion_module.reset_id_log()
            last_time_stamp = timestamp
            fusionTrkHist = TrackHistory(max_frames=7)

        # calculate time between two frames
        time_lag = (timestamp - last_time_stamp)
        last_time_stamp = timestamp

        # Get LiDAR detection for current frame
        det = detections[token]
        det.sort(reverse=True, key=lambda box:box['detection_score'])

        nmsDelay = 0
        # NMS algo from PolyMOT (Better performance)
        if cfg["DETECTION"]["use_nms"]:
            lsit_det, np_dets = dictdet2array(det, 'translation', 'size', 'velocity', 'rotation',
                                            'detection_score', 'detection_name')
            if len(np_dets) != 0:
                box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
                assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners)
                tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners}
                keep = blend_nms(box_infos=tmp_infos, metrics='iou_3d', thre=cfg["DETECTION"]["nms_th"])
                keep_num = len(keep)
            # corner case, no det left
            else: keep = keep_num = 0
            det = [det[i] for i in keep] if keep_num != 0 else []
        # My old NMS algoirthm
        # if cfg["DETECTION"]["use_nms"]:
        #     (det, _), nmsDelay = cal_func_time(nms, boxes=det, iou_th=cfg["DETECTION"]["nms_th"])

        # Get Radar targets
        radar_pc = dataset.get_key_radar_pc(token)

        # Radar segmentation with LiDAR detection
        det_for_viz = deepcopy(det)
        for obj in det_for_viz:
            obj['size'][0], obj['size'][1] = expand_ratio[0] * obj['size'][0], expand_ratio[1] * obj['size'][1]

        segResult, segDelay = cal_func_time(
            radarSeg.run, 
            radarTargets=radar_pc, 
            lidarDet=det, 
            score_thr=cfg["SEGMENTOR"]["detection_th"], 
            dist_th=cfg["SEGMENTOR"]["dist_th"],
            ego_pose=ego_pose,
        )
        radarSeg.reset()
        segResultList = []
        for point in segResult:
            seg = {
                "sample_token": token,
                "translation": list(point[:3]),
                "velocity": list(point[3:5]),
                "segment_id": int(point[5]),
                "category": int(point[6]),
            }
            segResultList.append(seg)
        nusc_seg_result["results"].update({token: deepcopy(segResultList)})

        # Tracking for LiDAR and Radar respectively
        lidar_trks, LtrkDelay = cal_func_time(lidar_tracker.step_centertrack, results=det, time_lag=time_lag)
        temp_lidar_trks = []
        lidar_active_trks = []
        for trk in lidar_trks:
            if 'active' in trk and trk['active'] < cfg["LIDAR_TRACKER"]["min_hits"]:
                continue
            
            trk['tracking_score'] = trk['detection_score']
            lidar_active_trks.append(trk)
            if trk['detection_name'] in NUSCENES_TRACKING_NAMES:
                nusc_trk = {
                    "sample_token": token,
                    "translation": list(trk['translation']),
                    "size": list(trk['size']),
                    "rotation": list(trk['rotation']),
                    "velocity": list(trk['velocity']),
                    "tracking_id": str(trk['tracking_id']),
                    "tracking_name": trk['detection_name'],
                    "tracking_score": trk['tracking_score'],
                }
                temp_lidar_trks.append(nusc_trk)

        nusc_LiDAR_trk["results"].update({token: deepcopy(temp_lidar_trks)})

        # Format radarSeg to object tracking type and perform tracking
        radarObjs = radar_tracker.formatForRadarSeg(segResult)
        radar_trks, RtrkDelay = cal_func_time(radar_tracker.step_centertrack, results=radarObjs, time_lag=time_lag)
        temp_radar_trks = []
        radar_active_trks = []
        for trk in radar_trks:
            if 'active' in trk and trk['active'] < cfg["RADAR_TRACKER"]["min_hits"]:
                continue

            trk['size'] = [1.0, 1.0, 1.0]    # pseudo size
            trk['tracking_score'] = trk['detection_score']
            radar_active_trks.append(trk)
            if trk['detection_name'] in RADAR_TRACKING_NAMES:
                nusc_trk = {
                    "sample_token": token,
                    "translation": list(trk['translation']),
                    "size": list(trk['size']),
                    "rotation": list(trk['rotation']),
                    "velocity": list(trk['velocity']),
                    "tracking_id": str(trk['tracking_id']),
                    "tracking_name": trk['detection_name'],
                    "tracking_score": trk['tracking_score'],
                }
                temp_radar_trks.append(nusc_trk)

        nusc_Radar_trk["results"].update({token: deepcopy(temp_radar_trks)})

        # Fusion module (ID arrangement and score update)
        fusion_trks = fusion_module.fuse(deepcopy(lidar_active_trks), deepcopy(radar_active_trks))
        temp_fusion_trks = []
        fusion_active_trks = []
        for trk in fusion_trks:
            trk['tracking_score'] = trk['detection_score']

            if 'active' in trk and trk['active'] < cfg["FUSION"]["min_hits"]:
                continue

            if trk['detection_name'] in NUSCENES_TRACKING_NAMES:
                nusc_trk = {
                    "sample_token": token,
                    "translation": list(trk['translation']),
                    "size": list(trk['size']),
                    "rotation": list(trk['rotation']),
                    "velocity": list(trk['velocity']),
                    "tracking_id": str(trk['tracking_id']),
                    "tracking_name": trk['detection_name'],
                    "tracking_score": trk['tracking_score'],
                }
                temp_fusion_trks.append(nusc_trk)

            fusion_active_trks.append(trk)

        nusc_fusion_trk["results"].update({token: deepcopy(temp_fusion_trks)})
        fusionTrkHist.update(deepcopy(fusion_active_trks))

        # Vizsualize (realtime)
        if args.viz:

            trans = dataset.get_4f_transform(ego_pose, inverse=True)
            viz_start = time.time()
            trackViz.draw_ego_car(img_src="/data/car1.png")
            trackViz.draw_radar_seg(radarSeg=segResult, trans=trans, **cfg["VISUALIZER"]["radarSeg"])
            trackViz.draw_det_bboxes(nusc_det=det_for_viz, trans=trans, **cfg["VISUALIZER"]["detBox"])
            trackViz.draw_det_bboxes(nusc_det=lidar_active_trks, trans=trans, **cfg["VISUALIZER"]["trkBox"])
            trackViz.draw_det_bboxes(radar_active_trks, trans, **cfg["VISUALIZER"]["radarTrkBox"])
            trackViz2.draw_ego_car(img_src="/data/car1.png")
            trackViz2.draw_radar_seg(segResult, trans, **cfg["VISUALIZER"]["radarSeg"])
            trackViz2.draw_det_bboxes(fusion_active_trks, trans, **cfg["VISUALIZER"]["fusionBox"])
            # trackViz2.draw_det_bboxes(radar_active_trks, trans, **cfg["VISUALIZER"]["radarTrkBox"])
            trackViz2.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
            # trackViz3.draw_ego_car(img_src="/data/car1.png")
            # trackViz3.draw_radar_seg(segResult, trans, **cfg["VISUALIZER"]["radarSeg"])
            k = len(fusionTrkHist)
            # for trks in fusionTrkHist.get_history():
            #     k -= 1
            #     alpha = 1.0 - 0.9 * k / len(fusionTrkHist)
            #     trackViz3.draw_det_bboxes(trks, trans, **cfg["VISUALIZER"]["trkHist"], alpha=alpha)
            trackViz.show()
            trackViz2.show()
            # trackViz3.show()
            viz_end = time.time()
            if args.show_delay:
                print(f"nms delay: {nmsDelay / 1e-3: .2f} ms")
                print(f"viz delay:{(viz_end - viz_start) / 1e-3: .2f} ms")

        pbar.update(1)

    pbar.close()

    # Close vizualization
    cv2.destroyAllWindows()

    # calculate computation time
    end = time.time()
    second = (end - start)
    speed = len_frames / second
    print("======")
    print("The speed is {} FPS".format(speed))
    print("tracking results have {} frames".format(len(nusc_fusion_trk["results"])))

    nusc_fusion_trk["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }
    nusc_LiDAR_trk["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    nusc_Radar_trk["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }
    nusc_seg_result["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }

    # write result file
    lidar_res_path = os.path.join(root_path, 'lidar_tracking_res.json')
    radar_res_path = os.path.join(root_path, 'radar_tracking_res.json')
    seg_res_path = os.path.join(root_path, 'radar_seg_res.json')
    with open(lidar_res_path, 'w') as f:
        json.dump(nusc_LiDAR_trk, f)
    with open(radar_res_path, 'w') as f:
        json.dump(nusc_Radar_trk, f)
    with open(track_res_path, 'w') as f:
        json.dump(nusc_fusion_trk, f)
    with open(seg_res_path, 'w') as f:
        json.dump(nusc_seg_result, f)
    print(f"tracking results write to {track_res_path}\n")

    # Evaluation
    if args.evaluate:
        print("======")
        print("Start evaluating tracking results")
        print(f"output directory: {cfg['EVALUATE']['out_dir']}")
        cfg["EVALUATE"]["res_path"] = track_res_path
        nusc_eval(**cfg["EVALUATE"])


if __name__ == "__main__":
    main(get_parser())