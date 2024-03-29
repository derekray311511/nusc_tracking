from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import Box
from email.header import Header
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from collections import deque
from copy import deepcopy
from multiprocessing import Pool

import sensor_msgs.point_cloud2 as pcl2
import numpy as np
import math
import os
import cv2
import rospy
import json
import argparse
import sys
import tf
import time

GT_CATEGORIES = [
    'vehicle.car', 'vehicle.motorcycle', 'vehicle.bicycle', 'vehicle.trailer',
    'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck', 
    'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.wheelchair', 
    'human.pedestrian.stroller', 'human.pedestrian.stroller', 'human.pedestrian.personal_mobility', 
    'human.pedestrian.police_officer', 'human.pedestrian.construction_worker', 
]

SHOW_CATEGORIES = [
    'bicycle',
    'motorcycle',
    'pedestrian',
    'bus',
    'car',
    'trailer',
    'truck',
]

def stack_pointclouds(read_data_func, frames, idx, stack_num):
    pcs = []
    for i in range(stack_num):
        if idx - i < 0:
            break
        pcs.append(read_data_func(frames[idx-i]['token']))
        if frames[idx-i]['first']:
            break
    stacked_pc = np.concatenate(pcs, axis=0)
    return stacked_pc

def stack_bboxes(read_data_func, frames, idx, stack_num, data_type, th):
    '''
    Order: [idx, idx-1, idx-2,...]
    '''
    frames_bboxes = []
    for i in range(stack_num):
        if idx - i < 0:
            break
        frames_bboxes.append(read_data_func(frames[idx-i]['token'], data_type, th))
        if frames[idx-i]['first']:
            break
    return frames_bboxes

def stack_radar_trks(read_data_func, frames, idx, stack_num):
    frames_trks = []
    for i in range(stack_num):
        if idx - i < 0:
            break
        try:
            frames_trks.append(read_data_func(frames[idx-i]['token']))
        except: # Temp
            frames_trks.append([])
        if frames[idx-i]['first']:
            break
    return frames_trks

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

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
    x y z w -> roll pitch yaw \n
    Convert a quaternion into euler angles (roll, pitch, yaw) \n
    roll is rotation around x in radians (counterclockwise) \n
    pitch is rotation around y in radians (counterclockwise) \n
    yaw is rotation around z in radians (counterclockwise) \n
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

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,   0],
                     [s,   c,   0],
                     [0,   0,   1]])

def get_3d_box(center, box_size, heading_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:heading_angle
        box_size: tuple of (l,w,h)
        : rad scalar, clockwise from pos z axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    Rz = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(Rz, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

''' =============== This part is usage for Trajectories =============== '''
class Object():
    def __init__(self, center, velocity):
        self.locations = deque(maxlen = 16)
        self.locations.appendleft(center)
        self.velocity = velocity

    def update(self, center, velocity):
        if center is not None:
            self.locations.appendleft(center)
            self.velocity = velocity

class Trajectory:
    def __init__(self, frames_centers, radar_type=False) -> None:
        self.centers = {}
        self.vels = {}
        self.trackers = {}
        for center_list in reversed(frames_centers):
            self.current_exist_ids = []
            self.save_path(center_list, radar_type)

    def save_path(self, center_list, radar_type=False):
        '''Save center list of objects

        Input: N * [x y z dx dy dz heading, vx, vy, id, name, score] for `bbox trackers`
        Input: N * [x y vx vy id] for `radar trackers type`
        Return: N * objects' paths
        '''
        if not radar_type:
            for obj in center_list:
                id = obj[9]
                self.centers[id] = obj[:2]
                self.vels[id] = obj[7:9]
                self.current_exist_ids.append(id)
        else:
            for obj in center_list:
                id = obj[4]
                self.centers[id] = obj[:2]
                self.vels[id] = obj[2:4]
                self.current_exist_ids.append(id)

        for track_id in self.centers:
            if track_id in self.trackers:  
                self.trackers[track_id].update(self.centers[track_id], self.vels[track_id])
            else:   
                self.trackers[track_id] = Object(self.centers[track_id], self.vels[track_id])

        for track_id in self.trackers:    
            if track_id not in self.centers:
                self.trackers[track_id].update(None, None)

        return self.trackers
''' =================================================================== '''

class show_keys:
    def __init__(self) -> None:
        self.font                   = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10,500)
        self.fontScale              = 3
        self.fontColor              = (255,255,255)
        self.thickness              = 5
        self.lineType               = cv2.LINE_AA

    def add_text(self, img, info=None):
        text = ['ESC', 'A', 'D', 'SPACE']
        pos = [(220, 100), (170, 300), (370, 300), (150, 500)]
        if info is not None:
            for i in range(len(info)):
                text.append(info[i])
                pos.append((5, 590 - 25*(len(info)-i-1)))
        for i in range(len(text)):
            if info is not None and i >= 4:
                fontScale = 0.6
                fontColor = (255,207,48)
                thickness = 1
            else:
                fontScale = self.fontScale
                fontColor = self.fontColor
                thickness = self.thickness
            cv2.putText(
                img,
                text[i], 
                pos[i], 
                self.font, 
                fontScale,
                fontColor,
                thickness,
                self.lineType,
            )
        return img

    def add_rect(self, img):
        points = [(140, 400), (460, 525)]
        cv2.rectangle(img, points[0], points[1], (255, 0, 0), 2)
        return img

class dataset:
    def __init__(
        self,
        dataroot='data/nuscenes',
        version='v1.0-trainval',
        split='val',
        detection_path='data//detection_result.json',
        track1_res_path='data/track_results/tracking_result.json', 
        track2_res_path=None, 
        frame_meta_path='data/frames_meta.json',
        radar_trk_path=None, 
        category_names=None,
    ):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.dataroot = dataroot
        if split == 'val':
            self.scene_names = splits.val
        elif split == 'train':
            self.scene_names = splits.train
        else:
            sys.exit(f"split {split} not support yet.")
        self.detections = self.load_detections(detection_path)
        self.tracklets1 = self.load_tracklets(track1_res_path)
        self.tracklets2 = self.load_tracklets(track2_res_path)
        self.frames = self.load_frames_meta(frame_meta_path)
        self.radar_trks = self.load_radar_trk(radar_trk_path)
        self.category_names = category_names
        self.imgs = {}

    def load_detections(self, path):
        if path is None or path == "None":
            return None
        with open(path, 'rb') as f:
            detections = json.load(f)['results']
        return detections

    def load_tracklets(self, path):
        if path is None or path == "None":
            return None
        with open(path, 'rb') as f:
            tracklets = json.load(f)['results']
        return tracklets

    def load_frames_meta(self, path):
        with open(path, 'rb') as f:
            frames = json.load(f)['frames']
        return frames

    def load_radar_trk(self, path):
        if path is None or path == "None":
            return None
        with open(path, 'rb') as f:
            data = json.load(f)
            try:
                radar_trks = data['key_results']
            except: # Old version
                radar_trks = data['results']
        return radar_trks

    def read_lidar_pc(self, token):
        sample_record = self.nusc.get('sample', token)
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        point_cloud = np.fromfile(os.path.join(self.dataroot, lidar_record['filename']), dtype=np.float32).reshape(-1,5)
        point_cloud = np.array(point_cloud[:, [0,1,2]])
        point_cloud = self.pc2world(point_cloud, token, 'LIDAR_TOP', inverse=False)
        return point_cloud

    def read_radar_pc(self, token):
        name_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        sample_record = self.nusc.get('sample', token)
        point_cloud = []
        for name in name_list:
            radar_record = self.nusc.get('sample_data', sample_record['data'][name])
            points = RadarPointCloud.from_file(os.path.join(self.dataroot, radar_record['filename'])).points.T
            points = points[:, [0, 1, 2, 8, 9]] # x, y, z, vx_comp, vy_comp
            points = self.pc2world(points, token, name, inverse=False)
            point_cloud.append(points)
        point_cloud = np.concatenate(point_cloud, axis=0)
        return point_cloud

    def get_4f_transform(self, pose, inverse=False):
        return transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=inverse)

    def pc2world(self, pointcloud, token, name='LIDAR_TOP', inverse=False):
        '''
        Input pointcloud shape: (n, 3) or (n, 5)
        '''
        pointcloud = np.array(pointcloud)
        if pointcloud.shape[1] == 3:
            use_vel = False
        if pointcloud.shape[1] == 5:
            use_vel = True

        sample_record = self.nusc.get('sample', token)
        sensor_record = self.nusc.get('sample_data', sample_record['data'][name])
        ego_pose = self.nusc.get('ego_pose', sensor_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', sensor_record['calibrated_sensor_token'])
        sensor2car = self.get_4f_transform(cs_record, inverse=inverse)
        car2world = self.get_4f_transform(ego_pose, inverse=inverse)

        trans = pointcloud[:, :3]
        if use_vel:
            vel = pointcloud[:, 3:5]
            vel = np.hstack([vel, np.zeros((vel.shape[0], 1))])
        if not inverse:
            new_trans = (car2world[:3, :3].dot((sensor2car[:3, :3].dot(trans.T).T + sensor2car[:3, 3]).T).T + car2world[:3, 3])
            if use_vel:
                new_vel = car2world[:3, :3].dot(sensor2car[:3, :3].dot(vel.T)).T
                new_trans = np.hstack([new_trans, new_vel[:, :2]])
        elif inverse:
            new_trans = (sensor2car[:3, :3].dot((car2world[:3, :3].dot(trans.T).T + car2world[:3, 3]).T).T + sensor2car[:3, 3])
            if use_vel:
                new_vel = sensor2car[:3, :3].dot(car2world[:3, :3].dot(vel.T)).T
                new_trans = np.hstack([new_trans, new_vel[:, :2]])

        return new_trans

    def get_bbox_result(self, token, data_type='detection', th=0):
        """Get tracking result for bounding box on specific sample token.

        data_type should be ['detection', 'track']

        (x, y, z) is the box center.  
        (vx, vy) is velocity based on world coordinates.

        Return: N * [x y z dx dy dz heading, vx, vy, id, name, score] or 
        Return: N * [x y z dx dy dz heading, vx, vy, name, score]
        """
        if data_type == 'detection':
            bboxes = self.detections[token] if self.detections is not None else []
        elif data_type == 'track1':
            bboxes = self.tracklets1[token] if self.tracklets1 is not None else []
        elif data_type == 'track2':
            bboxes = self.tracklets2[token] if self.tracklets2 is not None else []
        else:
            sys.exit(f"Wrong data type {data_type}!")
        
        new_boxes = []
        for bbox in bboxes:
            x, y, z = bbox['translation']
            vx, vy = bbox['velocity']
            Quaternion = q_to_xyzw(bbox['rotation'])
            roll, pitch, yaw = euler_from_quaternion(Quaternion)
            if (self.tracklets2 is not None) and (self.tracklets1 is not None) and data_type == 'track1':
                dx, dy, dz = bbox['size'][0]-0.5, bbox['size'][1]-0.5, bbox['size'][2]-0.5
            else:
                dx, dy, dz = bbox['size']
            # recentering for visualization # to be delete
            z = z + dz/2

            if data_type in ['track1', 'track2']:
                if bbox['tracking_score'] < th: continue
                if (self.category_names is not None) and (bbox['tracking_name'] not in self.category_names): continue
                id = bbox['tracking_id']
                name = bbox['tracking_name']
                score = bbox['tracking_score']
                new_boxes.append([x, y, z, dx, dy, dz, yaw, vx, vy, id, name, score])
            else:
                if bbox['detection_score'] < th: continue
                if (self.category_names is not None) and (bbox['detection_name'] not in self.category_names): continue
                name = bbox['detection_name']
                score = bbox['detection_score']
                new_boxes.append([x, y, z, dx, dy, dz, yaw, vx, vy, name, score])

        return new_boxes

    def get_gt_bbox(self, token):
        '''
        Return: N * [x y z dx dy dz heading]
        '''
        sample_record = self.nusc.get('sample', token)
        bboxes = []
        for annotation_token in sample_record['anns']:
            annotation = self.nusc.get('sample_annotation', annotation_token)
            if annotation['category_name'] in GT_CATEGORIES:
                x, y, z = annotation['translation']
                dx, dy, dz = annotation['size']
                Quaternion = q_to_xyzw(annotation['rotation'])
                roll, pitch, yaw = euler_from_quaternion(Quaternion)
                yaw = yaw - np.pi / 2
                bboxes.append([x, y, z, dx, dy, dz, yaw])
        return bboxes

    def get_radar_trks(self, token):
        '''
        Return: N * [x, y, vx, vy, id]
        '''
        if self.radar_trks is None:
            return []
        points = []
        for point in self.radar_trks[token]:
            x, y = point['translation']
            vx, vy = point['velocity']
            id = point['tracking_id']
            points.append([x, y, vx, vy, id])
        return points

    def get_cam_imgs(self, token):
        '''Get nusc raw camera images

        Return images dictionary by sensor name
        '''
        name_list = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]
        sample_record = self.nusc.get('sample', token)
        imgs = {}
        for sensor in name_list:
            filename = self.nusc.get('sample_data', sample_record['data'][sensor])['filename']
            filename = os.path.join(self.dataroot, filename)
            imgs.update({sensor: cv2.imread(filename)})
        return imgs

    def world2cam4f(self, token, cam_name):
        sample_record = self.nusc.get('sample', token)
        sd_record = self.nusc.get('sample_data', sample_record['data'][cam_name])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        viewpad = np.eye(4)
        viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic

        # world to ego
        world2car = self.get_4f_transform(pose_record, inverse=True)
        # ego to camera
        car2cam = self.get_4f_transform(cs_record, inverse=True)
        # camera to image
        cam2img = viewpad
        # world to image
        world2img = cam2img @ car2cam @ world2car
        return world2img

    def cam_with_box(self, token, data_type='detection', th=0.1, id_color=True, RGB_color=None, prev_imgs=None):
        '''Get nusc camera images with bboxes

        Return images dictionary by sensor name
        '''
        if data_type == 'detection':
            base_color = np.array([66, 135, 245]) / 255.0; id_color = False
        elif data_type == 'track1':
            base_color = np.array([250, 185, 85]) / 255.0
        elif data_type == 'track2':
            base_color = np.array([55, 250, 143]) / 255.0
        else:
            sys.exit(f"Wrong data type {data_type}!")

        if RGB_color is not None:
            color = np.array(RGB_color, dtype=np.float)
            base_color = np.clip(color, a_min=0, a_max=1)
        
        color = deepcopy(base_color)
        bboxes_corners = []
        ids = []
        bboxes = self.get_bbox_result(token, data_type, th)
        for bbox in bboxes:
            x, y, z, w, l, h, yaw, vx, vy = bbox[:9]
            corners = get_3d_box((x, y, z), (l, w, h), yaw)
            bboxes_corners.append(corners)
            if data_type in ['track1', 'track2']:
                ids.append(int(bbox[9]))
        if data_type in ['track1', 'track2']:
            ids = np.array(ids, dtype=np.int)
        bboxes_corners = np.array(bboxes_corners)

        sample_record = self.nusc.get('sample', token)
        name_list = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
        ]
        imgs = {} if prev_imgs is None else prev_imgs
        for sensor in name_list:
            if prev_imgs is None:
                sd_record = self.nusc.get('sample_data', sample_record['data'][sensor])
                filename = sd_record['filename']
                filename = os.path.join(self.dataroot, filename)
                img = cv2.imread(filename)
            else:
                img = imgs[sensor]

            transform = self.world2cam4f(token, sensor)

            num_bboxes = bboxes_corners.shape[0]
            coords = np.concatenate(
                [bboxes_corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
            )
            transform = deepcopy(transform).reshape(4, 4)
            coords = coords @ transform.T
            coords = coords.reshape(-1, 8, 4)

            indices = np.all(coords[..., 2] > 0, axis=1)
            coords = coords[indices]
            # labels = labels[indices]
            if data_type in ['track1', 'track2']:
                obj_ids = ids[indices]

            indices = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[indices]
            # labels = labels[indices]
            if data_type in ['track1', 'track2']:
                obj_ids = obj_ids[indices]

            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]

            coords = coords[..., :2].reshape(-1, 8, 2)

            for index in range(coords.shape[0]):
                # Set colors (BGR)
                if id_color:
                    id = obj_ids[index]
                    color[2] = int((base_color[0] * 255 + (id) * 15) % 155 + 100)   # R 100~255
                    color[1] = int((base_color[1] * 255 - (id) * 20) % 155 + 100)   # G 100~255
                    color[0] = 100                                                  # B 100
                else:
                    color = (base_color * 255)
                    color = [int(color[2]), int(color[1]), int(color[0])]  # RGB to BGR

                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        img,
                        coords[index, start].astype(np.int),
                        coords[index, end].astype(np.int),
                        color=color,
                        thickness=4,
                        lineType=cv2.LINE_AA,
                    )
            imgs.update({sensor: img})
        return imgs


class pub_data:
    def __init__(self):
        self.br = tf.TransformBroadcaster()
        self.lidar_pub = rospy.Publisher("lidar_pointcloud", PointCloud2, queue_size=10)
        self.radar_pub = rospy.Publisher("radar_pointcloud", PointCloud2, queue_size=10)
        self.front_cam_pub = rospy.Publisher("front_camera", Image, queue_size=10)
        self.back_cam_pub = rospy.Publisher("back_camera", Image, queue_size=10)
        self.det_pub = rospy.Publisher("detection", MarkerArray, queue_size=10)
        self.trk_pub = rospy.Publisher("tracking", MarkerArray, queue_size=10)
        self.gt_pub = rospy.Publisher("groundtruth", MarkerArray, queue_size=10)
        self.traj_pub = rospy.Publisher('trajectory', MarkerArray, queue_size=10)
        self.radar_traj_pub = rospy.Publisher('radar_trajectory', MarkerArray, queue_size=10)
        self.radar_raw_vel_pub = rospy.Publisher('radar_raw_vel', MarkerArray, queue_size=10)
        self.trk_markerArray = MarkerArray()
        self.det_markerArray = MarkerArray()
        self.gt_markerArray = MarkerArray()
        self.traj_markerArray = MarkerArray()
        self.radar_traj_markerArray = MarkerArray()
        self.radar_raw_vel_markerArray = MarkerArray()

    def pub_pc(self, point_cloud, name='lidar', viz_r_vel=False):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'
        if name == 'lidar':
            self.lidar_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud))
        if name == 'radar':
            self.radar_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:, :3]))
            if not viz_r_vel:
                return
            self.pub_radar_vel(point_cloud)

    def broadcast(self, nusc_data, token):
        sample_record = nusc_data.get('sample', token)
        LIDAR_record = nusc_data.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = nusc_data.get('ego_pose', LIDAR_record['ego_pose_token'])
        self.br.sendTransform(ego_pose['translation'], q_to_xyzw(ego_pose['rotation']), rospy.Time.now(), 'ego_pose', 'world')

    def pub_radar_vel(self, pointcloud, RGB_color=None):
        '''
        Publish raw radar points' velocity
        '''
        markerArray = self.radar_raw_vel_markerArray
        color = np.array([239, 41, 41]) / 255.0
        if RGB_color is not None:
            color = np.array(RGB_color, dtype=np.float)
            color = np.clip(color, a_min=0, a_max=1)

        for i in range(len(pointcloud)):
            x, y, z, vx, vy = pointcloud[i]
            if np.sqrt(vx**2 + vy**2) < 0.1:
                continue
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.ns = 'radar_raw_velocity'

            marker.id = i
            marker.action = Marker.ADD
            marker.type = Marker.ARROW
            marker.lifetime = rospy.Duration(0)

            point = Point()
            point.x = x
            point.y = y
            point.z = z
            marker.points.append(point)
            point = Point()
            point.x = x + vx
            point.y = y + vy
            point.z = z
            marker.points.append(point)

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.scale.x = 0.25
            marker.scale.y = 0.5

            markerArray.markers.append(marker)
        self.radar_raw_vel_pub.publish(markerArray)

    def pub_bboxes(
        self, 
        bboxes, 
        namespace='detection', 
        show_id=True, 
        show_vel=True, 
        show_score=True, 
        id_color=False, 
        RGB_color=None,
        ):
        if namespace == 'detection':
            color = np.array([66, 135, 245]) / 255.0; id_color = False
        elif namespace == 'track1':
            color = np.array([250, 185, 85]) / 255.0
        elif namespace == 'track2':
            color = np.array([55, 250, 143]) / 255.0
        elif namespace == 'gt':
            color = np.array([201, 65, 44]) / 255.0
            self.draw_cube(bboxes, color, namespace)
            return
        else:
            sys.exit("Wrong namespace of pub_bboxes!")

        if RGB_color is not None:
            color = np.array(RGB_color, dtype=np.float)
            color = np.clip(color, a_min=0, a_max=1)

        bboxes_corners = []
        for bbox in bboxes:
            x, y, z, w, l, h, yaw, vx, vy = bbox[:9]
            center = [x, y, z]
            velocity = [vx, vy]
            if namespace == 'detection':
                sub = 0.5   # Subtract the bbox size for visualization
                corners = get_3d_box((x, y, z), (l-sub, w-sub, h-sub), yaw)
                name = bbox[9]
                score = bbox[10]
                bbox = [corners, center, velocity, name, score]
            if namespace in ['track1', 'track2']:
                corners = get_3d_box((x, y, z), (l, w, h), yaw)
                name = bbox[10]
                id = bbox[9]
                score = bbox[11]
                bbox = [corners, center, velocity, name, id, score]
            bboxes_corners.append(bbox)
        self.draw_bboxes(bboxes_corners, color, namespace, show_id, show_vel, show_score, id_color)

    def pub_cam(self, imgs):
        img_list = []
        for i, (name, img) in enumerate(imgs.items()):
            cv2.putText(img, name, (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 3, (20, 100, 160), 3, cv2.LINE_AA)
            img_list.append(img)
        img_front = np.hstack([img_list[0], img_list[1], img_list[2]])
        img_back = np.hstack([img_list[3], img_list[4], img_list[5]])
        bridge = CvBridge()
        self.front_cam_pub.publish(bridge.cv2_to_imgmsg(img_front, "bgr8"))
        self.back_cam_pub.publish(bridge.cv2_to_imgmsg(img_back, "bgr8"))

    def draw_cube(self, bboxes, RGBcolor, namespace='gt', show_id=False, show_vel=False):

        for i in range(len(bboxes)):
            q = get_quaternion_from_euler(0, 0, bboxes[i][6])
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace

            marker.id = i
            marker.action = Marker.ADD
            marker.type = Marker.CUBE
            marker.lifetime = rospy.Duration(0)

            marker.pose.position.x = bboxes[i][0]
            marker.pose.position.y = bboxes[i][1]
            marker.pose.position.z = bboxes[i][2]
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = bboxes[i][3]
            marker.scale.y = bboxes[i][4]
            marker.scale.z = bboxes[i][5]

            marker.color.r = RGBcolor[0]
            marker.color.g = RGBcolor[1]
            marker.color.b = RGBcolor[2]
            marker.color.a = 0.4

            self.gt_markerArray.markers.append(marker)
        
        self.gt_pub.publish(self.gt_markerArray)

    def draw_bboxes(
        self, 
        bboxes, 
        RGBcolor, 
        namespace='detection', 
        show_id=True, 
        show_vel=True, 
        show_score=True,
        id_color=False,
        ):
        duration = 0

        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        base_color = deepcopy(RGBcolor)

        if namespace in ['track1', 'track2']:
            markerArray = self.trk_markerArray
        elif namespace =='detection':
            markerArray = self.det_markerArray

        print(f"{namespace}-bboxes:{len(bboxes)}")
            
        for obid in range(len(bboxes)):
            ob = bboxes[obid][0]
            center = bboxes[obid][1]
            velocity = bboxes[obid][2]
            name = bboxes[obid][3]
            if namespace in ['track1', 'track2']:
                id = bboxes[obid][4]
                score = bboxes[obid][5]
            elif namespace =='detection':
                score = bboxes[obid][4]

            # Draw with id color
            if id_color:
                RGBcolor[0] = ((base_color[0] * 255 + int(id) * 15) % 155 + 100) / 255.0    # 100~255
                RGBcolor[1] = ((base_color[1] * 255 - int(id) * 20) % 155 + 100) / 255.0    # 100~255
                RGBcolor[2] = 100 / 255.0

            # Draw bbox lines
            bbox_points_set = []
            for i in range(8):
                bbox_points_set.append(Point(ob[i][0], ob[i][1], ob[i][2]))

            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace

            marker.id = obid
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.lifetime = rospy.Duration(duration)

            marker.color.r = RGBcolor[0]
            marker.color.g = RGBcolor[1]
            marker.color.b = RGBcolor[2]
            marker.color.a = 1.0
            marker.scale.x = 0.3
            marker.points = []

            for line in lines:
                marker.points.append(bbox_points_set[line[0]])
                marker.points.append(bbox_points_set[line[1]])

            markerArray.markers.append(marker)

            # Draw bbox id if exist
            if namespace in ['track1', 'track2'] and show_id:
                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = rospy.Time.now()
                marker.ns = namespace + '_id'

                marker.id = obid
                marker.action = Marker.ADD
                marker.type = Marker.TEXT_VIEW_FACING
                marker.lifetime = rospy.Duration(duration)

                marker.color.r = base_color[0]
                marker.color.g = base_color[1]
                marker.color.b = base_color[2]
                marker.color.a = 1.0
                marker.scale.x = 2.0
                marker.scale.y = 2.0
                marker.scale.z = 2.0
                marker.pose.position.x = center[0]
                marker.pose.position.y = center[1]
                marker.pose.position.z = center[2]
                marker.text = id

                markerArray.markers.append(marker)

            # Draw bbox score
            if show_score:
                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = rospy.Time.now()
                marker.ns = namespace + '_score'

                marker.id = obid
                marker.action = Marker.ADD
                marker.type = Marker.TEXT_VIEW_FACING
                marker.lifetime = rospy.Duration(duration)

                marker.color.r = base_color[0]
                marker.color.g = base_color[1]
                marker.color.b = 200 / 255.0
                marker.color.a = 1.0
                marker.scale.x = 2.0
                marker.scale.y = 2.0
                marker.scale.z = 2.0
                marker.pose.position.x = center[0] + 1.0
                marker.pose.position.y = center[1] + 1.0
                marker.pose.position.z = center[2] + 2.0
                marker.text = str(np.round(score, 2))

                markerArray.markers.append(marker)

            # Draw bbox velcoity
            if show_vel:
                if np.sqrt(velocity[0]**2 + velocity[1]**2) < 0.05:
                    continue
                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = rospy.Time.now()
                marker.ns = namespace + '_velocity'

                marker.id = obid
                marker.action = Marker.ADD
                marker.type = Marker.ARROW
                marker.lifetime = rospy.Duration(duration)

                point = Point()
                point.x = center[0]
                point.y = center[1]
                point.z = center[2]
                marker.points.append(point)
                point = Point()
                point.x = center[0] + velocity[0]
                point.y = center[1] + velocity[1]
                point.z = center[2]
                marker.points.append(point)

                marker.color.r = RGBcolor[0]
                marker.color.g = RGBcolor[1]
                marker.color.b = RGBcolor[2]
                marker.color.a = 1.0
                marker.scale.x = 0.4
                marker.scale.y = 0.8

                markerArray.markers.append(marker)

        if namespace == 'detection':
            self.det_pub.publish(markerArray)
        elif namespace in ['track1', 'track2']:
            self.trk_pub.publish(markerArray)

    def pub_trajectory(self, traj, id_color=False, namespace='track1', RGB_color=None):
        if namespace == 'track1':
            base_color = np.array([250, 185, 85]) / 255.0
        elif namespace == 'track2':
            base_color = np.array([55, 250, 143]) / 255.0
        else:
            base_color = np.array([214, 91, 66]) / 255.0

        if RGB_color is not None:
            color = np.array(RGB_color, dtype=np.float) / 255.0
            color = np.clip(color, a_min=0, a_max=1)
        else:
            color = deepcopy(base_color)

        current_exist_ids = traj.current_exist_ids
        trackers = traj.trackers

        duration = 0
        for track_id in current_exist_ids:
            if len(trackers[track_id].locations) <= 1:
                continue
            # Draw with id color
            if id_color:
                color[0] = ((base_color[0] * 255 + int(track_id) * 15) % 155 + 100) / 255.0    # 100~255
                color[1] = ((base_color[1] * 255 - int(track_id) * 20) % 155 + 100) / 255.0    # 100~255
                color[2] = 100 / 255.0
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace + '_trajectory'
            marker.id = int(track_id)
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(duration)
            marker.type = Marker.LINE_STRIP
            marker.pose.orientation.w = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.6
            marker.scale.x = 0.2
            marker.points.clear()
            marker.points = []
            for p in trackers[track_id].locations:
                marker.points.append(Point(p[0], p[1], 0))
            if namespace in ['track1', 'track2']:
                self.traj_markerArray.markers.append(marker)
            elif namespace == 'radar_trk':
                self.radar_traj_markerArray.markers.append(marker)

            # Draw velocity for radar trackers
            if namespace != 'radar_trk':
                continue
            velocity = trackers[track_id].velocity
            center = trackers[track_id].locations[0]
            if np.sqrt(velocity[0]**2 + velocity[1]**2) < 0.05:
                continue
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = rospy.Time.now()
            marker.ns = namespace + '_velocity'
            marker.id = int(track_id)
            marker.action = Marker.ADD
            marker.type = Marker.ARROW
            marker.lifetime = rospy.Duration(duration)
            point = Point()
            point.x = center[0]
            point.y = center[1]
            point.z = 0.0
            marker.points.append(point)
            point = Point()
            point.x = center[0] + velocity[0]
            point.y = center[1] + velocity[1]
            point.z = 0.0
            marker.points.append(point)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.scale.x = 0.4
            marker.scale.y = 0.6
            self.radar_traj_markerArray.markers.append(marker)

        if namespace in ['track1', 'track2']:
            self.traj_pub.publish(self.traj_markerArray)
        elif namespace == 'radar_trk':
            self.radar_traj_pub.publish(self.radar_traj_markerArray)

    def clear_markers(self):
        markerArray_list = [
            self.gt_markerArray, 
            self.det_markerArray, 
            self.trk_markerArray, 
            self.traj_markerArray, 
            self.radar_traj_markerArray,
            self.radar_raw_vel_markerArray,
        ]
        publisher_list = [
            self.gt_pub, 
            self.det_pub, 
            self.trk_pub, 
            self.traj_pub, 
            self.radar_traj_pub,
            self.radar_raw_vel_pub,    
        ]
        for markerArray in markerArray_list:
            for marker in markerArray.markers:
                # marker.color.a = 0
                marker.lifetime = rospy.Duration(0.1)
        # Clear markers in rviz
        for (pub, markerArray) in zip(publisher_list, markerArray_list):
            pub.publish(markerArray)
        # Clear markers in backend
        for markerArray in markerArray_list:
            markerArray.markers.clear()

from threading import Thread
class CustomThread(Thread):
    def __init__(self, func, args=()):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.value = None
 
    # function executed in a new thread
    def run(self):
        self.value = self.func(*self.args)

def read_all_data(data, frames, idx, args):
    '''Load raw data and bounding boxes
    
    Return dict of data, including keys 'lidar_pc', 'radar_pc', 'frames_det', 'frames_trk', 'gt', 'trajectory', 'cam_imgs'
    '''
    token = frames[idx]['token']
    lidar_pc = stack_pointclouds(data.read_lidar_pc, frames, idx, args.lidar_stack)
    radar_pc = stack_pointclouds(data.read_radar_pc, frames, idx, args.radar_stack)
    frames_det = stack_bboxes(data.get_bbox_result, frames, idx, args.det_bbox_stack, data_type='detection', th=args.vis_th)
    frames_trk1 = stack_bboxes(data.get_bbox_result, frames, idx, args.trk_bbox_stack, data_type='track1', th=args.vis_th)
    frames_trk2 = stack_bboxes(data.get_bbox_result, frames, idx, args.trk_bbox_stack, data_type='track2', th=args.vis_th)
    gt = data.get_gt_bbox(token)
    trajectory1 = Trajectory(frames_trk1) # trk_bbox_stack number
    trajectory2 = Trajectory(frames_trk2)
    if args.vis_img_bbox:
        cam_imgs = data.cam_with_box(token, data_type='track1', th=args.vis_th, id_color=args.id_color)
        cam_imgs = data.cam_with_box(token, data_type='track2', th=args.vis_th, id_color=args.id_color, prev_imgs=cam_imgs)
    else:
        cam_imgs = data.get_cam_imgs(token)
    result_data = {
        'lidar_pc': lidar_pc,
        'radar_pc': radar_pc,
        'frames_det': frames_det,
        'frames_trk1': frames_trk1, 
        'frames_trk2': frames_trk2, 
        'gt': gt,
        'trajectory1': trajectory1,
        'trajectory2': trajectory2,
        'cam_imgs': cam_imgs,
    }
    if args.viz_radar_trks:
        frames_radar_trk = stack_radar_trks(data.get_radar_trks, frames, idx, args.trk_bbox_stack)
        radar_trajectory = Trajectory(frames_radar_trk, radar_type=True) # trk_bbox_stack number
        result_data.update({
            'frames_radar_trk': frames_radar_trk,
            'radar_trajectory': radar_trajectory,
        })
    return result_data

def publish_all_result(publisher, data, frames, idx, result_data, args):
    token = frames[idx]['token']
    if args.id_color:
        trk_show = [False, True, True, True]
    else:
        trk_show = [True, True, False, False]
    publisher.clear_markers()
    publisher.pub_pc(result_data['lidar_pc'], name='lidar')
    publisher.pub_pc(result_data['radar_pc'], name='radar', viz_r_vel=args.viz_r_vel)
    publisher.broadcast(data.nusc, token)
    publisher.pub_cam(result_data['cam_imgs'])
    publisher.pub_bboxes(result_data['gt'], namespace='gt', show_id=False, show_vel=False, show_score=False, id_color=False)
    publisher.pub_bboxes(result_data['frames_det'][0], namespace='detection', show_id=False, show_vel=False, show_score=False, id_color=False)
    publisher.pub_bboxes(result_data['frames_trk1'][0], namespace='track1', show_id=trk_show[0], show_vel=trk_show[1], show_score=trk_show[2], id_color=trk_show[3])
    publisher.pub_bboxes(result_data['frames_trk2'][0], namespace='track2', show_id=trk_show[0], show_vel=trk_show[1], show_score=trk_show[2], id_color=trk_show[3])
    publisher.pub_trajectory(result_data['trajectory1'], id_color=args.id_color, namespace='track1')
    publisher.pub_trajectory(result_data['trajectory2'], id_color=args.id_color, namespace='track2')
    if 'radar_trajectory' in result_data and args.viz_radar_trks:
        publisher.pub_trajectory(result_data['radar_trajectory'], id_color=False, namespace='radar_trk', RGB_color=[42, 213, 222])

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--dataroot", type=str, default='/home/Student/Tracking/data/nuscenes')
    parser.add_argument("--detection_path", type=str, default='/home/Student/Tracking/data/detection_result.json')
    parser.add_argument("--track1_res_path", type=str, default=None)
    parser.add_argument("--track2_res_path", type=str, default=None)
    parser.add_argument("--frames_meta_path", type=str, default='/home/Student/Tracking/data/frames_meta.json')
    parser.add_argument("--vis_gt", type=int, default=1)
    parser.add_argument("--pub_rate", type=float, default=4)
    parser.add_argument("--vis_th", type=float, default=0)
    parser.add_argument("--vis_img_bbox", type=int, default=1)
    parser.add_argument("--init_idx", type=int, default=0)
    parser.add_argument("--id_color", type=int, default=1)
    # Stack multi-frame info
    parser.add_argument("--lidar_stack", type=int, default=2)
    parser.add_argument("--radar_stack", type=int, default=1)
    parser.add_argument("--det_bbox_stack", type=int, default=1)
    parser.add_argument("--trk_bbox_stack", type=int, default=5)
    # Radar viz
    parser.add_argument("--radar_trk_path", type=str, default=None)
    parser.add_argument("--viz_radar_trks", type=int, default=0)
    parser.add_argument("--viz_r_vel", type=int, default=0, help="Visualize the raw radar velocity or not, use 1 or 0")
    # multi-thread
    parser.add_argument("--multi_thread", type=int, default=1)
    return parser

def main(parser):
    args, opts = parser.parse_known_args()

    # Load data, detection, track
    if args.split in ['train', 'val']:
        version = 'v1.0-trainval'
    else:
        sys.exit(f"Split {args.split} not support yet.")
        
    data = dataset(
        dataroot=args.dataroot,
        version=version,
        split=args.split,
        detection_path=args.detection_path,
        track1_res_path=args.track1_res_path,
        track2_res_path=args.track2_res_path,
        frame_meta_path=args.frames_meta_path,
        radar_trk_path=args.radar_trk_path,
        category_names=SHOW_CATEGORIES,
    )
    key_shower = show_keys()
    publisher = pub_data()

    # Ros initialization
    rospy.init_node('pub_nusc', anonymous=True)
    duration = 1.0 / args.pub_rate
    rate = rospy.Rate(args.pub_rate)

    # Visualization
    frames = data.frames
    idx = args.init_idx
    max_idx = len(frames) - 1
    last_idx = idx

    auto_playing_mode = False
    init = True
    
    while (1):
        
        if rospy.is_shutdown():
            rospy.loginfo('shutdown')
            break

        if init:
            cv2.namedWindow("keys", cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar('Frame', 'keys', 0, max_idx, lambda x: None)
            cv2.setTrackbarPos('Frame', 'keys', idx)
        else:
            cv2.setTrackbarPos('Frame', 'keys', idx)
            if auto_playing_mode:
                idx += 1
                key = cv2.waitKey(int(duration * 1000))
                if key == 27 or idx > max_idx: # esc
                    break
                if key == 32: # space
                    auto_playing_mode = not auto_playing_mode
            else:
                key = cv2.waitKey(0)
            
                if key == 100 or key == 83: # d
                    idx += 1
                if key == 97 or key == 81: # a
                    idx -= 1
                if key > max_idx:
                    idx -= 1
                    print("Out of length!")
                if key == 13:
                    idx = cv2.getTrackbarPos('Frame', 'keys')
                    init = True # Get certain frame's data
                if key == 27: # esc
                    break
                if idx < 0:
                    idx = 0
                if key == 32: # space
                    auto_playing_mode = not auto_playing_mode

        token = frames[idx]['token']
        timestamp = frames[idx]['timestamp']
        info = '{}-{}'.format(timestamp, token)

        ctl_img = np.zeros((600, 600, 3))
        ctl_img = key_shower.add_text(ctl_img, [info])
        if auto_playing_mode:
            ctl_img = key_shower.add_rect(ctl_img)
        cv2.imshow('keys', ctl_img)

        if init:
            result_data = read_all_data(data, frames, idx, args)
            init = False
        else:
            if idx == last_idx - 1 or (idx == 1 and last_idx == 0) or (idx == max_idx-1 and last_idx == max_idx):
                choose = 0
            elif idx == last_idx + 1:
                choose = 1
            else:
                continue
            result_data = result_data_temp[choose]
        last_idx = idx

        if not args.multi_thread:
            # Publish raw data and bounding boxes
            start = time.time()
            start1 = time.time()
            publish_all_result(publisher, data, frames, idx, result_data, args)
            end1 = time.time()

            start2 = time.time()
            result_data_temp = []   # list for multi-idx results
            # Load raw data and bounding boxes for next and prev frame
            if idx-1 >= 0:
                result_data_temp.append(read_all_data(data, frames, idx-1, args))
            if idx+1 <= max_idx:
                result_data_temp.append(read_all_data(data, frames, idx+1, args))
            end2 = time.time()
            end = time.time()
            print(f"Pub time: {end1-start1:.4f} sec")
            print(f"Load time: {end2-start2:.4f} sec")
        else:
            # Publish raw data and bounding boxes
            start = time.time()
            pub_thread = Thread(target=publish_all_result, args=(publisher, data, frames, idx, result_data, args))
            pub_thread.start()

            result_data_temp = []   # list for multi-idx results
            # Load raw data and bounding boxes for next and prev frame
            if idx-1 >= 0:
                load_thread1 = CustomThread(read_all_data, (data, frames, idx-1, args))
                load_thread1.start()
            if idx+1 <= max_idx:
                load_thread2 = CustomThread(read_all_data, (data, frames, idx+1, args))
                load_thread2.start()

            if idx-1 >= 0:
                load_thread1.join()
                result_data_temp.append(load_thread1.value)
            if idx+1 <= max_idx:
                load_thread2.join()
                result_data_temp.append(load_thread2.value)
            pub_thread.join()
            end = time.time()
            
        print(f"Pub and Preload time: {end-start:.4f} sec")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(parser())
    