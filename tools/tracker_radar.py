import copy
from copy import deepcopy
from curses import KEY_PPAGE
from nis import match
import math

from cv2 import threshold
import tf

import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
    'construction_vehicle',
    'barrier',
    'traffic_cone',
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

RADAR_FUSION_NAMES = [
    # 'bus', 
    'car', 
    'motorcycle',
    'trailer', 
    'truck', 
    'construction_vehicle', 
]

RADAR_UNCERTAINTY_PARAMETER = 2
RADAR_CLS_VELOCITY_ERROR = {key: value * RADAR_UNCERTAINTY_PARAMETER for key, value in NUSCENE_CLS_VELOCITY_ERROR.items()}
# RADAR_CLS_VELOCITY_ERROR = {
#     'car': 6,
#     'truck': 8,
#     'bus': 11,
#     'trailer': 4,
#     'pedestrian': 2,
#     'motorcycle': 8,
#     'bicycle': 5,
#     'construction_vehicle': 2,
#     'barrier': 2,
#     'traffic_cone': 2,
# }

print(f"RADAR_CLS_VELOCITY_ERROR = \n{RADAR_CLS_VELOCITY_ERROR}")

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

    positions1_cat = np.array([index['label_preds'] for index in positions1_data], np.int32)  # M pos1 labels
    positions2_cat = np.array([index['label_preds'] for index in positions2_data], np.int32)  # N pos2 labels
    max_diff = np.array([self.velocity_error[box['detection_name']] for box in positions2_data], np.float32)

    if len(positions1) > 0:  # NOT FIRST FRAME
        dist = (((positions1.reshape(1, -1, 2) - positions2.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
        dist = np.sqrt(dist)  # absolute distance in meter
        invalid = ((dist > max_diff.reshape(N, 1)) + (
                positions2_cat.reshape(N, 1) != positions1_cat.reshape(1, M))) > 0
        dist = dist + invalid * 1e18
        if self.hungarian:
            reshaped_matched_indices = []
            dist[dist > 1e18] = 1e18
            matched_indices = linear_sum_assignment(deepcopy(dist))
            for i in range(len(matched_indices[0])):
                reshaped_matched_indices.append([matched_indices[0][i], matched_indices[1][i]])
            matched_indices = np.array(reshaped_matched_indices, np.int32).reshape(-1, 2)
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

def euler_to_quaternion(roll, pitch, yaw):
    """
    Return [qx, qy, qz, qw]
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def Nuscenes_get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qw, qx, qy, qz: The orientation in quaternion [w,x,y,z] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qw, qx, qy, qz]

def check_point_in_box(pts, box):
    """
    pts[x,y,z]
    box[cx,cy,cz,dx,dy,dz,heading]
    """
    shift_x = pts[0] - box[0]
    shift_y = pts[1] - box[1]
    shift_z = pts[2] - box[2]
    cos_a = math.cos(box[6])
    sin_a = math.sin(box[6])
    dx, dy, dz = box[3], box[4], box[5]
    local_x = shift_x * cos_a + shift_y * sin_a
    local_y = shift_y * cos_a - shift_x * sin_a
    if abs(shift_z) > dz / 2.0 or abs(local_x) > dx / 2.0 or abs(local_y) > dy / 2.0:
        return False
    else:    
        return True

def dist2d(p1, p2):
    """
    Get the distance between 2 2d/3d points p1, p2
    """
    return np.linalg.norm(p1-p2)

# Give radar trackers objects' id
def update_radar_id(self, track, radar_trackers, match):
    """
    Add an object id to each radar tracker due to the bbox\n
    Return radar_trackers
    """
    count = 0
    temp = []
    track_point = track['ct']
    Quaternion = track['rotation']
    heading = get_yaw_from_quaternion(Quaternion)
    box = [track_point[0], track_point[1], 0, track['size'][0] + 0.0, track['size'][1] + 1.0, track['size'][2] + 2.0, heading]
    for radar_tracker in radar_trackers:
        point = np.pad(radar_tracker['translation'], (0, 1), mode='constant', constant_values=0)    # Zero padding [x, y] -> [x, y, 0]
        if match:
            if check_point_in_box(point, box):
                radar_tracker['obj_id'] = track['tracking_id']
                radar_tracker['label_preds'] = track['label_preds']
                count += 1
        else:
            if check_point_in_box(point, box) and radar_tracker['obj_id'] == -1:
                radar_tracker['obj_id'] = track['tracking_id']
                radar_tracker['label_preds'] = track['label_preds']
                count += 1
        # Remove outlier
        box_out = [track_point[0], track_point[1], 0, track['size'][0] + 1.0, track['size'][1] + 1.5, track['size'][2] + 2.0, heading]
        if (not check_point_in_box(point, box_out)) and radar_tracker['obj_id'] == track['tracking_id']:
            radar_tracker['obj_id'] = -1
            radar_tracker['label_preds'] = ''
        temp.append(radar_tracker)
    # if count != 0:
    #     print("{} radar points id set to obj".format(count))
    return temp

# Pass obj_id to next radar trackers
def pass_obj_id(prev_radar_trackers, radar_trackers):
    temp = []
    for radar_tracker in radar_trackers:
        for prev_tracker in prev_radar_trackers:
            if int(radar_tracker['tracking_id']) == int(prev_tracker['tracking_id']):
                radar_tracker['obj_id'] = prev_tracker['obj_id']
                radar_tracker['label_preds'] = prev_tracker['label_preds']
                break
        temp.append(radar_tracker)
    return temp

def pass_obj_id2(prev2_radar_trackers, radar_trackers):
    temp = []
    for radar_tracker in radar_trackers:
        # Only pass obj_id that is not set
        if radar_tracker['obj_id'] != -1:
            temp.append(radar_tracker)
            continue
        for prev_tracker in prev2_radar_trackers:
            if int(radar_tracker['tracking_id']) == int(prev_tracker['tracking_id']):
                radar_tracker['obj_id'] = prev_tracker['obj_id']
                radar_tracker['label_preds'] = prev_tracker['label_preds']
                break
        temp.append(radar_tracker)
    return temp

# Update bboxes' pose by matching radar_trackers' id
def update_by_radar_tracker(self, track, radar_trackers, min_score):
    """
    Return bbox center ('ct') or [x, y]
    """
    id = track['tracking_id']
    counter = 0
    center = np.array([0.0, 0.0])
    radar_min_score = min_score
    for radar_track in radar_trackers:
        radar_id = radar_track['obj_id']
        radar_cat = radar_track['label_preds']
        track_cat = track['detection_name']
        dist_th = RADAR_CLS_VELOCITY_ERROR[track_cat]
        if dist2d(track['KF'].x[:2], radar_track['translation']) > dist_th:
            continue
        if radar_id == id and (radar_cat == track['label_preds']):
            counter += 1
            center += radar_track['translation']
    if counter == 0:
        return track['ct'], counter, track['detection_score']
    else:
        # multiplication
        new_score = 1 - (1 - track['detection_score']) * (1 - radar_min_score)
        self.Update_score_by_radarTracker += 1
        return center / counter, counter, new_score

# For radar points
def update_score_with_radar(self, radar_data, track):
    """
    Return the score we want to add correspend with radar points match.
    """
    Use_velocity = True
    radar_min_score = 0.2
    radar_points_center = [0.0, 0.0]

    track_point = track['ct']
    Quaternion = track['rotation']
    heading = get_yaw_from_quaternion(Quaternion)
    box = [track_point[0], track_point[1], 0, track['size'][0] + 0.3, track['size'][1] + 0.3, track['size'][2] + 0.3, heading]
    # radar_points = np.array([radar_point['pose'] for radar_point in radar_data['points']], np.float32)  # M x 2

    match_count = 0
    count_th = 3

    if not Use_velocity:
        count_th = 2
        for data in radar_data['points']:
            point = np.concatenate((data['pose'], [0]))
            if check_point_in_box(point, box):
                match_count = match_count + 1
    else:
        for data in radar_data['points']:
            if data['vel'] < 1.0:
                continue
            point = np.concatenate((data['pose'], [0]))
            if check_point_in_box(point, box):
                match_count += 1
                radar_points_center += point[:2]
        if match_count != 0:
            radar_points_center = np.array(radar_points_center) / match_count

    if match_count >= count_th:
        self.Update_score_by_radarPoints += 1
        new_score = 1 - (1 - track['detection_score']) * (1 - radar_min_score)
        return new_score, match_count, radar_points_center
    else:
        new_score = track['detection_score']
        return new_score, match_count, radar_points_center

def update_heading_by_velocity(self, track):
    if self.tracker == 'PointTracker':
        offset = track['tracking'] * -1
    elif self.tracker == 'KF':
        offset = track['velocity']
    original_rot = track['rotation']
    Quaternion = shift_quaternion(original_rot)
    euler = tf.transformations.euler_from_quaternion(Quaternion)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    original_heading = yaw
    rad = np.arctan2(offset[1], offset[0])
    track['rotation'] = Nuscenes_get_quaternion_from_euler(roll, pitch, rad)
    return track

def update_function(self, track, m, loaded_model):
    if self.update_function == 'nn':
        inp = np.array([self.tracks[m[1]]['detection_score'], track['detection_score']])
        new = loaded_model(torch.Tensor(inp))
        track['detection_score'] = new.item()

    elif self.update_function == 'parallel_addition':
        track['detection_score'] = 1 - ((1 - track['detection_score']) * (
                1 - self.tracks[m[1]]['detection_score'])) / (
                                           (1 - track['detection_score']) + (
                                           1 - self.tracks[m[1]]['detection_score']))
    elif self.update_function == 'multiplication':
        track['detection_score'] = 1 - (1 - track['detection_score']) * \
                                   (1 - self.tracks[m[1]]['detection_score'])
    elif self.update_function == 'addition':
        track['detection_score'] += self.tracks[m[1]]['detection_score']
        track['detection_score'] = np.clip(track['detection_score'], a_min=0.0, a_max=1.0)
    elif self.update_function == 'max':
        track['detection_score'] = np.maximum(track['detection_score'],
                                              self.tracks[m[1]]['detection_score'])

    return track


# reshape hungarians output to match the greedy output shape
def reshape(hungarian):
    result = np.empty((0, 2), int)
    for i in range(len(hungarian[0])):
        result = np.append(result, np.array([[hungarian[0][i], hungarian[1][i]]]), axis=0)
    return result


class PubTracker(object):
    def __init__(
        self, 
        hungarian=False, 
        max_age=6, 
        active_th=1, 
        min_hits=1, 
        update_function=None,
        update_model=None,
        score_decay=0.15,
        deletion_th=0.0, 
        detection_th=0.0, 
        use_vel=True, 
        tracker='KF', 
        radar_fusion=True,
        ):
        self.tracker = 'PointTracker' if tracker is None else tracker
        self.hungarian = hungarian
        self.max_age = max_age
        self.min_hits = min_hits
        self.s_th = active_th  # activate threshold
        self.update_function = update_function
        self.update_model = update_model  # Update score with nn model if update_model is not None
        self.score_decay = score_decay
        self.det_th = detection_th  # detection threshold
        self.del_th = deletion_th  # deletion threshold
        self.use_vel = use_vel
        # Radar parameters
        self.radar_fusion = radar_fusion
        self.radar_active_th = 0.0
        self.Active_by_radarTracker_cat = np.zeros((len(RADAR_FUSION_NAMES)), dtype=np.int)
        self.Active_by_radarTracker = 0
        self.Update_score_by_radarTracker = 0

        print("=========================================")
        print("Tracker: {}".format(self.tracker))
        print("Use hungarian: {}".format(self.hungarian))
        print("Use velocity: {}".format(self.use_vel))
        print("Max age: {}".format(self.max_age))
        print("Min_hits: {}".format(self.min_hits))
        print("Detection th: {}\n".format(self.det_th))
        print("Update function: {}".format(self.update_function))
        print("Update model: {}".format(self.update_model))
        print("Score decay: {}".format(self.score_decay))
        print("=========================================")
        print("Radar fusion: {}".format(self.radar_fusion))
        if self.radar_fusion:
            print("Radar active th: {}".format(self.radar_active_th))
        print("=========================================")

        self.velocity_error = NUSCENE_CLS_VELOCITY_ERROR
        self.tracking_names = NUSCENES_TRACKING_NAMES
        self.id_count = 0
        self.tracks = []
        self.NofuseCamera_tracks = []
        self.prev_radar_trackers = []
        self.prev2_radar_trackers = []
        self.prev3_radar_trackers = []
        self.prev4_radar_trackers = []
        self.prev5_radar_trackers = []
        self.frame_count = 0

        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks = []
        self.NofuseCamera_tracks = []
        self.prev_radar_trackers = []
        self.prev2_radar_trackers = []
        self.prev3_radar_trackers = []
        self.prev4_radar_trackers = []
        self.prev5_radar_trackers = []
        self.frame_count = 0

    def step_centertrack(self, results, time_lag, radar_trackers):
        """
        computes connections between current resources with resources from older frames
        :param results: resources in one specific frame
        :param time_lag: time between two successive frame (difference in their timestamp)
        :param radar_trackers: radar points tracklet from frames
        :return: tracks: tracklets (detection + tracking id, age, activity) for one specific frame
        """

        self.frame_count += 1

        # =====================================================
        self.NofuseCamera_tracks = []
        if self.radar_fusion:
            # Set all obj_id to -1
            r_temp = []
            for r_tracker in radar_trackers:
                r_tracker['obj_id'] = -1
                r_tracker['label_preds'] = ''
                r_temp.append(r_tracker)
            radar_trackers = r_temp

            # Pass obj_id from previous radar_trackers
            if len(self.prev_radar_trackers) != 0:
                radar_trackers = pass_obj_id(self.prev_radar_trackers, radar_trackers)
            # Pass obj_id from prev2 to radar trackers
            if len(self.prev2_radar_trackers) != 0:
                radar_trackers = pass_obj_id2(self.prev2_radar_trackers, radar_trackers)
            # Pass obj_id from prev3 to radar trackers
            if len(self.prev3_radar_trackers) != 0:
                radar_trackers = pass_obj_id2(self.prev3_radar_trackers, radar_trackers)
            # Pass obj_id from prev4 to radar trackers
            if len(self.prev4_radar_trackers) != 0:
                radar_trackers = pass_obj_id2(self.prev4_radar_trackers, radar_trackers)
            # Pass obj_id from prev5 to radar trackers
            if len(self.prev5_radar_trackers) != 0:
                radar_trackers = pass_obj_id2(self.prev5_radar_trackers, radar_trackers)

        # =====================================================

        # if no detection in this frame, reset tracks list
        if len(results) == 0:
            self.tracks = []  # <-- however, this means, all tracklets are gone (i.e. 'died')
            return []

        # if any detection is found, ...
        else:
            temp = []
            for det in results:  # for each detection ...
                # filter out classes not evaluated for tracking
                if det['detection_name'] not in self.tracking_names:
                    continue
                # for all evaluated classes, extend with the following attributes
                det['ct'] = np.array(det['translation'][:2])  # ct: 2d centerpoint of one detection
                det['radar_track_num'] = 0
                if self.tracker == 'PointTracker':
                    det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
                # label_preds: class id (instead of class name)
                det['label_preds'] = self.tracking_names.index(det['detection_name'])
                temp.append(det)

            results = temp  # contains all extended resources

        N = len(results)  # number of resources in this frame
        M = len(self.tracks)  # number of tracklets
        ret = []  # initiate return value (will become the updated tracklets list)

        # if no tracklet exist just yet (i.e. processing the first frame)
        if M == 0:
            for result in results:  # for each (extended) detection
                # initiate new tracklet
                track = result
                self.id_count += 1
                # extend tracklet with the following attributes:
                track['tracking_id'] = self.id_count  # tracklet id
                track['age'] = 1  # how many frames without matching detection (i.e. inactivity)\
                track['active'] = self.min_hits  # currently matched? (start with 1)
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
                    track['KF'].x = np.hstack([track['ct'], np.array(track['velocity'][:2]), np.zeros(2)])
                    track['KF'].F = np.array([[1, 0, time_lag, 0, 0.5 * time_lag * time_lag, 0],
                                                [0, 1, 0, time_lag, 0, 0.5 * time_lag * time_lag],
                                                [0, 0, 1, 0, time_lag, 0],
                                                [0, 0, 0, 1, 0, time_lag],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]])
                    track['KF'].P *= 10
                    track['KF'].R *= 1
                ret.append(track)
            self.tracks = ret
            return ret

        # Processing from the second frame
        if self.tracker == 'PointTracker':
            # N X 2
            # dets: estmated 2d centerpoint of a detection in the previous frame (ct + expected offset)
            if N > 0:
                if 'tracking' in results[0]:
                    dets = np.array(
                        [det['ct'].astype(np.float32) + det['tracking'].astype(np.float32)
                        for det in results], np.float32)
                else:
                    dets = np.array(
                        [det['ct'] for det in results], np.float32)
            else:
                dets = np.array([], np.float32)
                
            tracks = np.array(
                [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        elif self.tracker == 'KF':
            if N > 0:
                dets = np.array(
                    [det['ct'] for det in results], np.float32)
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
        matching = comparing_positions(self, self.tracks, results, tracks, dets)
        matched, unmatched_trk, unmatched_det = matching[0], matching[1], matching[2]

        # add matches
        for m in matched:
            # initiate new tracklet (with three additional attributes)
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']  # tracklet id = id of matched trackled
            track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
            track['active'] = self.tracks[m[1]]['active'] + 1
            track['radar_track_num'] = 0
            if self.tracker == 'KF':
                track['KF'] = self.tracks[m[1]]['KF']
                if self.use_vel:
                    track['KF'].update(z=np.hstack([track['ct'], np.array(track['velocity'][:2])]))
                else:
                    track['KF'].update(z=track['ct'])
                track['ct'][0] = track['KF'].x[0]
                track['ct'][1] = track['KF'].x[1]
                track['velocity'][0] = track['KF'].x[2]
                track['velocity'][1] = track['KF'].x[3]

            # update detection score
            self.tracks[m[1]]['detection_score'] = np.clip(self.tracks[m[1]]['detection_score'] - self.score_decay, a_min=0.0, a_max=1.0)
            if self.update_function is not None:
                track['detection_score'] = update_function(self, track, m, self.update_model)['detection_score']
            # update detection score by radar and update radar_id
            if self.radar_fusion and track['detection_name'] in RADAR_FUSION_NAMES:
                temp_ct, track['radar_track_num'], track['detection_score'] = update_by_radar_tracker(self, track, radar_trackers, 0.2) # 0.2
                radar_trackers = update_radar_id(self, track, radar_trackers, match=True)
            
            ret.append(track)

        # add unmatched resources as new 'born' tracklets
        for i in unmatched_det:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
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
                track['KF'].x = np.hstack([track['ct'], np.array(track['velocity'][:2]), np.zeros(2)])
                track['KF'].P *= 10
                track['KF'].R *= 1

            # uncomment these line and comment the line above if you want to make the experiments
            # in which with the resources are active only above a threshold
            if track['detection_score'] > self.det_th:
                track['active'] = 1
            else:
                track['active'] = 0
            ret.append(track)

        # still store unmatched tracks, however, we shouldn't output the object in current frame
        for i in unmatched_trk:
            track = self.tracks[i]

            # update score (only apply score decay)
            if self.update_function is not None:
                track['detection_score'] -= self.score_decay

            track['radar_track_num'] = 0
            ct = track['ct']

            # ================================================================
            if self.radar_fusion and track['detection_name'] in RADAR_FUSION_NAMES:
                # Update pose with radar_trackers (based on obj id)
                temp_ct, track['radar_track_num'], track['detection_score'] = update_by_radar_tracker(self, track, radar_trackers, 0.3)
                if self.tracker == 'PointTracker':
                    if track['radar_track_num'] != 0:
                        track['ct'] = temp_ct
                        track['tracking']  = ct - temp_ct
                        track['velocity'][:2] = track['tracking'] * -1 / time_lag
                        # print("tracking:", track['tracking'])
                        # print(f"id: {track['tracking_id']}, tracking: {track['tracking']}, radar_num: {track['radar_track_num']}, ct: {ct}, track[ct]: {track['ct']}")
                        track = update_heading_by_velocity(self, track)
                    else:
                        if 'tracking' in track:
                            offset = track['tracking'] * -1  # move forward
                            track['ct'] = ct + offset
                    # radar_trackers = update_radar_id(self, track, radar_trackers)
                    # if track['tracking_id'] == int(40):
                    #     print(f"1-id: {track['tracking_id']}, [x, y]: {track['ct']}, vel: {track['velocity'][:2]}")
                elif self.tracker == 'KF':
                    if ct[0] != temp_ct[0] or ct[1] != temp_ct[1]:
                        if self.use_vel:
                            track['KF'].update(z=np.hstack([temp_ct, np.array((temp_ct - ct) / time_lag)]))
                        else:
                            track['KF'].update(z=temp_ct)
                        track['ct'][0] = track['KF'].x[0]
                        track['ct'][1] = track['KF'].x[1]
                        track['velocity'][0] = track['KF'].x[2]
                        track['velocity'][1] = track['KF'].x[3]
                        track = update_heading_by_velocity(self, track)
                    else:
                        track['ct'][0] = track['KF'].x[0]
                        track['ct'][1] = track['KF'].x[1]
                        track['velocity'][0] = track['KF'].x[2]
                        track['velocity'][1] = track['KF'].x[3]
                    # radar_trackers = update_radar_id(self, track, radar_trackers)

            else:
                if 'tracking' in track and self.tracker == 'PointTracker':
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                elif self.tracker == 'KF':
                    track['ct'][0] = track['KF'].x[0]
                    track['ct'][1] = track['KF'].x[1]
                    track['velocity'][0] = track['KF'].x[2]
                    track['velocity'][1] = track['KF'].x[3]

            # ================================================================

            # keep tracklet if score is above threshold AND age is not too high
            if track['age'] < self.max_age and track['detection_score'] > self.del_th:
                if self.radar_fusion:
                    self.s_th = self.radar_active_th
                    if track['radar_track_num'] == 0:
                        track['age'] += 1
                    else:
                        track['age'] = 1
                else:
                    track['age'] += 1

                if self.radar_fusion:
                    if track['detection_score'] > self.s_th and track['radar_track_num'] != 0:   # Only active when score > th and having same id radar tracker
                        track['active'] += 1
                        radar_trackers = update_radar_id(self, track, radar_trackers, match=False)
                        if track['radar_track_num'] > 0:
                            # print(f"unMatched trk radar track num: {track['radar_track_num']}, score: {track['detection_score']}")
                            self.Active_by_radarTracker += 1
                            self.Active_by_radarTracker_cat[RADAR_FUSION_NAMES.index(track['detection_name'])] += 1
                    else:
                        track['active'] = 0
                else:
                    if track['detection_score'] > self.s_th:
                        track['active'] += 1
                    else:
                        track['active'] = 0

                ret.append(track)


        for i in range(len(ret)):
            ret[i]['translation'][:2] = ret[i]['ct']
            # if ret[i]['tracking_id'] == int(40):
            #     print(f"2-id: {ret[i]['tracking_id']}, [x, y]: {ret[i]['translation']}, vel: {ret[i]['velocity'][:2]}")

        # ================================================================
        if self.radar_fusion:
            # Do NMS (non maximum suppression)
            # nms_threshold = 0.5
            # boxes = tracks_to_boxes(ret)
            # ret, new_boxes = NMS(ret, boxes, nms_threshold)

            if self.frame_count > 4:
                self.prev5_radar_trackers = self.prev4_radar_trackers
            if self.frame_count > 3:
                self.prev4_radar_trackers = self.prev3_radar_trackers
            if self.frame_count > 2:
                self.prev3_radar_trackers = self.prev2_radar_trackers
            if self.frame_count > 1:
                self.prev2_radar_trackers = self.prev_radar_trackers
            self.prev_radar_trackers = radar_trackers

        # ================================================================

        self.tracks = ret + self.NofuseCamera_tracks
        return ret
