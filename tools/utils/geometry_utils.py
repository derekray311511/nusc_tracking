import numpy as np
import math
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

def eucl2D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    
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

def det_transform(objects, ego_trans):
    ret = []
    for object in objects:
        trans = np.array(object['translation'])
        vel = np.array([object['velocity'][0], object['velocity'][1], 0.0])
        rot = quaternion_rotation_matrix(object['rotation'])
        trans = np.hstack([rot, trans.reshape(-1, 1)])
        trans = np.vstack([trans, np.array([0, 0, 0, 1])]).reshape(-1, 4)
        vel = vel.reshape(-1, 1)
        new_trans = ego_trans @ trans
        new_vel = ego_trans[:3, :3] @ vel
        object['translation'] = new_trans[:3, 3].ravel().tolist()
        object['rotation'] = q_to_wxyz(R.from_matrix(new_trans[:3, :3]).as_quat())
        object['velocity'] = new_vel.ravel()[:2]
        ret.append(object)
    return ret

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
