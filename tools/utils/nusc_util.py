from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy
from .geometry_utils import pc2world
from .utils import mkdir_or_exist
from .utils import get_current_datetime
import json
import os
import time
import argparse
import numpy as np
import sys

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]

RADAR_NAMES = [
    'RADAR_FRONT', 
    # 'RADAR_FRONT_LEFT', 
    # 'RADAR_FRONT_RIGHT', 
    # 'RADAR_BACK_LEFT', 
    # 'RADAR_BACK_RIGHT'
]

class nusc_dataset:
    def __init__(self, cfg):
        self.nusc_version = cfg["nusc_version"]
        self.dataroot = cfg["dataroot"]
        self.gt = self.load_groundTruth(cfg["groundTruth_path"])
        self.split = cfg["split"]
        self.categories = NUSCENES_TRACKING_NAMES
        self.det_res = self.load_detections(cfg["lidar_det_path"])
        self.frames = self.load_frames_meta(cfg["frames_meta_path"])
        self.lidar_PCs = self.load_lidar_pc(cfg["lidar_pc_path"])
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

    def load_lidar_pc(self, path):
        if path is None or path == "None":
            print(f"lidar pointcloud path is None!")
            return
        with open(path, 'rb') as f:
            lidar_PCs = json.load(f)['lidar_PCs']
        return lidar_PCs
    
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
        temp_radar_PCs_token = []
        for k, radar_pc in self.radar_PCs.items():
            if not radar_pc['is_key_frame']:
                temp_radar_PCs_token.append(k)
                continue
            sample_token = radar_pc['sample_token']
            key_radar_pc = {
                'token': sample_token,
                'radar_token': k,
                'prev_radar_tokens': temp_radar_PCs_token,
                'ego_pose_token': radar_pc['ego_pose_token'],
                'points': radar_pc['points'],
            }
            key_radar_PCs.update({sample_token: key_radar_pc})
            temp_radar_PCs_token = []
            count += 1
        print(f"{count} key radar_PCs loaded")
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

    def get_radar_pcs(self, key_token, max_stack=7):
        radar_pcs = []
        stack = 1
        for token in reversed(self.key_radar_PCs[key_token]['prev_radar_tokens']):
            if stack >= max_stack:
                break
            radar_pcs += self.radar_PCs[token]['points']
            stack += 1
        radar_pcs += self.key_radar_PCs[key_token]['points']
        return radar_pcs

    def get_lidar_pc(self, token):
        return self.lidar_PCs[token]

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

    def add_key_frames_info(self, frames):
        first_frame_token = None
        first_frame_idx = None
        print("Adding key frame info...")
        for idx in tqdm(range(len(frames))):
            frame = frames[idx]
            token = frame['token']
            if frame['first']:
                first_frame_token = token
                first_frame_idx = idx
            frame.update({
                'first_frame_token': first_frame_token,
                'first_frame_idx': first_frame_idx,
            })
        print("Done.")
        return frames

class nusc_data:
    def __init__(
        self, 
        nusc_version='v1.0-trainval',
        dataroot=None,
        verbose=True,
    ):
        self.nusc_version = nusc_version
        self.dataroot = dataroot
        self.nusc = NuScenes(self.nusc_version, self.dataroot, verbose=verbose)
        self.trackCat = [
            'bicycle',
            'motorcycle',
            'pedestrian',
            'bus',
            'car',
            'trailer',
            'truck',
        ]
        self.catMapping = {
            'animal': None,
            'human.pedestrian.personal_mobility': None,
            'human.pedestrian.stroller': None,
            'human.pedestrian.wheelchair': None,
            'movable_object.barrier': None,
            'movable_object.debris': None,
            'movable_object.pushable_pullable': None,
            'movable_object.trafficcone': None,
            'static_object.bicycle_rack': None,
            'vehicle.emergency.ambulance': None,
            'vehicle.emergency.police': None,
            'vehicle.construction': None,
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck',
        }
        
    def load_radarPC(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        radar_sample_data = {
            'RADAR_FRONT': None, 
            'RADAR_FRONT_LEFT': None, 
            'RADAR_FRONT_RIGHT': None, 
            'RADAR_BACK_LEFT': None, 
            'RADAR_BACK_RIGHT': None,
        }
        for sensor_name in RADAR_NAMES:
            sample_data = self.nusc.get('sample_data', sample['data'][sensor_name])
            radar_sample_data.update({sensor_name: sample_data})
        
        # Get radar pointcloud
        point_cloud = []
        for sensor_name in RADAR_NAMES:
            if radar_sample_data[sensor_name] is None:
                continue
            ego_pose_token = radar_sample_data[sensor_name]['ego_pose_token']
            calib_token = radar_sample_data[sensor_name]['calibrated_sensor_token']
            points = RadarPointCloud.from_file(os.path.join(self.dataroot, radar_sample_data[sensor_name]['filename'])).points.T
            points = points[:, [0, 1, 2, 8, 9]] # x, y, z, vx_comp, vy_comp
            points = pc2world(self.nusc, points, ego_pose_token, calib_token, sensor_name, inverse=False)
            point_cloud.append(points)
        point_cloud = np.concatenate(point_cloud, axis=0).tolist()

        return point_cloud

if __name__=="__main__":
    print("None")