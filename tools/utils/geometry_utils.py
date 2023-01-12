import numpy as np
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def get_4f_transform(pose, inverse=False):
    return transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=inverse)

def pc2world(nusc, pointcloud, ego_pose_token, calib_token, name='LIDAR_TOP', inverse=False):
    '''
    Input pointcloud shape: (n, 3) or (n, 5)
    '''
    pointcloud = np.array(pointcloud)
    if pointcloud.shape[1] == 3:
        use_vel = False
    if pointcloud.shape[1] == 5:
        use_vel = True

    ego_pose = nusc.get('ego_pose', ego_pose_token)
    cs_record = nusc.get('calibrated_sensor', calib_token)
    sensor2car = get_4f_transform(cs_record, inverse=inverse)
    car2world = get_4f_transform(ego_pose, inverse=inverse)

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
