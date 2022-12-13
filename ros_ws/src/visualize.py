from math import atan2, sqrt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from email.header import Header
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from copy import deepcopy

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
            text.append(info)
            pos.append((5, 590))
        for i in range(len(text)):
            if info is not None and i == len(text)-1:
                fontScale = 0.65
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
        track_res_path='data/track_results/tracking_result.json', 
        frame_meta_path='data/frames_meta.json',
    ):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.dataroot = dataroot
        if split == 'val':
            self.scene_names = splits.val
        elif split == 'train':
            self.scene_names = splits.train
        else:
            sys.exit(f"split {split} not support yet.")
        self.detections = self.load_detections(detection_path)['results']
        self.tracklets = self.load_tracklets(track_res_path)['results']
        self.frames = self.load_frames_meta(frame_meta_path)['frames']

    def load_detections(self, path):
        with open(path, 'rb') as f:
            detections = json.load(f)
        return detections

    def load_tracklets(self, path):
        with open(path, 'rb') as f:
            tracklets = json.load(f)
        return tracklets

    def load_frames_meta(self, path):
        with open(path, 'rb') as f:
            frames = json.load(f)
        return frames

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

        [x, y, z, dx, dy, dz, heading, vx, vy, id, name] in each box.
        (x, y, z) is the box center.  
        (vx, vy) is velocity based on world coordinates.

        Return: N * [x y z dx dy dz heading, vx, vy, id, name]
        """
        if data_type == 'detection':
            bboxes = self.detections[token]
        elif data_type == 'track':
            bboxes = self.tracklets[token]
        else:
            sys.exit(f"Wrong data type {data_type}!")
        
        new_boxes = []
        for bbox in bboxes:
            x, y, z = bbox['translation']
            dx, dy, dz = bbox['size']
            vx, vy = bbox['velocity']
            Quaternion = q_to_xyzw(bbox['rotation'])
            roll, pitch, yaw = euler_from_quaternion(Quaternion)
            if data_type == 'track':
                if bbox['tracking_score'] < th: continue
                id = bbox['tracking_id']
                name = bbox['tracking_name']
                score = bbox['tracking_score']
                new_boxes.append([x, y, z, dx, dy, dz, yaw, vx, vy, id, name, score])
            else:
                if bbox['detection_score'] < th: continue
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

    def get_cam_imgs(self, token):
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
        self.trk_markerArray = MarkerArray()
        self.det_markerArray = MarkerArray()
        self.gt_markerArray = MarkerArray()

    def pub_pc(self, point_cloud, name='lidar'):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'
        if name == 'lidar':
            self.lidar_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud))
        if name == 'radar':
            self.radar_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud))

    def broadcast(self, nusc_data, token):
        sample_record = nusc_data.get('sample', token)
        LIDAR_record = nusc_data.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = nusc_data.get('ego_pose', LIDAR_record['ego_pose_token'])
        self.br.sendTransform(ego_pose['translation'], q_to_xyzw(ego_pose['rotation']), rospy.Time.now(), 'ego_pose', 'world')

    def pub_bboxes(self, bboxes, namespace='detection', show_id=True, show_vel=True, show_score=True, id_color=False):
        if namespace == 'detection':
            color = np.array([66, 135, 245]) / 255.0; id_color = False
        elif namespace == 'track':
            color = np.array([237, 185, 52]) / 255.0
        elif namespace == 'gt':
            color = np.array([201, 65, 44]) / 255.0
            self.draw_cube(bboxes, color, namespace)
            return
        else:
            sys.exit("Wrong namespace of pub_bboxes!")

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
            if namespace == 'track':
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

        for obid in range(len(bboxes)):
            ob = bboxes[obid][0]
            center = bboxes[obid][1]
            velocity = bboxes[obid][2]
            name = bboxes[obid][3]
            if namespace == 'track':
                markerArray = self.trk_markerArray
                id = bboxes[obid][4]
                score = bboxes[obid][5]
            elif namespace =='detection':
                markerArray = self.det_markerArray
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
            marker.ns = 'bbox'

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
            if namespace == 'track' and show_id:
                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = rospy.Time.now()
                marker.ns = 'id'

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
                marker.ns = 'score'

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
                marker.ns = 'velocity'

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
        elif namespace == 'track':
            self.trk_pub.publish(self.trk_markerArray)

    def clear_markers(self):
        markerArray_list = [self.det_markerArray, self.trk_markerArray, self.gt_markerArray]
        for markerArray in markerArray_list:
            for marker in markerArray.markers:
                marker.color.a = 0
        # Clear markers in rviz
        self.gt_pub.publish(self.gt_markerArray)
        self.det_pub.publish(self.det_markerArray)
        self.trk_pub.publish(self.trk_markerArray)
        # Clear markers in backend
        for markerArray in markerArray_list:
            markerArray.markers.clear()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--dataroot", type=str, default='/home/Student/Tracking/data/nuscenes')
    parser.add_argument("--detection_path", type=str, default='/home/Student/Tracking/data/detection_result.json')
    parser.add_argument("--track_res_path", type=str, default='/home/Student/Tracking/data/track_results/tracking_result.json')
    parser.add_argument("--frames_meta_path", type=str, default='/home/Student/Tracking/data/frames_meta.json')
    parser.add_argument("--vis_gt", type=int, default=1)
    parser.add_argument("--pub_rate", type=float, default=4)
    parser.add_argument("--vis_th", type=float, default=0)
    parser.add_argument("--init_idx", type=int, default=0)
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
        track_res_path=args.track_res_path,
        frame_meta_path=args.frames_meta_path,
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

    auto_playing_mode = True
    init = True

    while (1):
        
        if rospy.is_shutdown():
            rospy.loginfo('shutdown')
            break

        token = frames[idx]['token']
        timestamp = frames[idx]['timestamp']
        info = '{}-{}'.format(timestamp, token)

        img = np.zeros((600, 600, 3))
        img = key_shower.add_text(img, info)
        if auto_playing_mode:
            img = key_shower.add_rect(img)
        cv2.imshow('keys', img)

        # Load raw data and bounding boxes
        lidar_pc = data.read_lidar_pc(token)
        radar_pc = data.read_radar_pc(token)[:, :3]
        cam_imgs = data.get_cam_imgs(token)
        det = data.get_bbox_result(token, data_type='detection', th=args.vis_th)
        trk = data.get_bbox_result(token, data_type='track', th=args.vis_th)
        gt = data.get_gt_bbox(token)

        # Publish raw data and bounding boxes
        publisher.clear_markers()
        publisher.pub_pc(lidar_pc, name='lidar')
        publisher.pub_pc(radar_pc, name='radar')
        publisher.broadcast(data.nusc, token)
        publisher.pub_cam(cam_imgs)
        publisher.pub_bboxes(det, namespace='detection', show_id=False, show_vel=False, show_score=False, id_color=False)
        publisher.pub_bboxes(trk, namespace='track', show_id=True, show_vel=True, show_score=False, id_color=False)
        publisher.pub_bboxes(gt, namespace='gt', show_id=False, show_vel=False, show_score=False, id_color=False)

        if init:
            cv2.waitKey(0)
            init = False
        else:
            if auto_playing_mode:
                idx += 1
                key = cv2.waitKey(int(duration * 1000))
                if key == 13 or key == 27 or idx > max_idx: # enter or esc
                    break
                if key == 32: # space
                    auto_playing_mode = not auto_playing_mode
            else:
                key = cv2.waitKey(0)
            
                if key == 100: # d
                    idx += 1
                if key == 97: # a
                    idx -= 1
                if key > max_idx:
                    idx -= 1
                    print("Out of length!")
                if key == 13 or key == 27: # enter or esc
                    break
                if idx < 0:
                    idx = 0
                if key == 32: # space
                    auto_playing_mode = not auto_playing_mode

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(parser())
