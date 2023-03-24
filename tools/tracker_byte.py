import copy

import numpy as np
import torch
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

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

def update_function(self, track, m, loaded_model):
    if self.score_update == 'nn':
        inp = np.array([self.tracks[m[1]]['detection_score'], track['detection_score']])
        new = loaded_model(torch.Tensor(inp))
        track['detection_score'] = new.item()

    elif self.score_update == 'parallel_addition':
        track['detection_score'] = 1 - ((1 - track['detection_score']) * (
                1 - self.tracks[m[1]]['detection_score'])) / (
                                           (1 - track['detection_score']) + (
                                           1 - self.tracks[m[1]]['detection_score']))
    elif self.score_update == 'multiplication':
        track['detection_score'] = 1 - (1 - track['detection_score']) * \
                                   (1 - self.tracks[m[1]]['detection_score'])
    elif self.score_update == 'addition':
        track['detection_score'] += self.tracks[m[1]]['detection_score']
        track['detection_score'] = np.clip(track['detection_score'], a_min=0.0, a_max=1.0)
    elif self.score_update == 'max':
        track['detection_score'] = np.maximum(track['detection_score'],
                                              self.tracks[m[1]]['detection_score'])

    return track

WAYMO_TRACKING_NAMES = [
    1,
    2,
    4,
]
WAYMO_CLS_VELOCITY_ERROR = {
    1: 2,
    2: 0.2,
    4: 0.5,
}


# reshape hungarians output to match the greedy output shape
def reshape(hungarian):
    result = np.empty((0, 2), int)
    for i in range(len(hungarian[0])):
        result = np.append(result, np.array([[hungarian[0][i], hungarian[1][i]]]), axis=0)
    return result


class PubTracker(object):
    def __init__(self, hungarian=False, max_age=6, noise=0.05, active_th=1, min_hits=1, score_update=None,
                 deletion_th=0.0, detection_th=0.0, dataset='Nuscenes', use_vel=False, tracker=None):
        self.tracker = 'PointTracker' if tracker is None else tracker
        self.hungarian = hungarian
        self.max_age = max_age
        self.min_hits = min_hits
        self.noise = noise
        self.s_th = active_th  # activate threshold
        self.score_update = score_update
        self.det_th = detection_th  # detection threshold
        self.del_th = deletion_th  # deletion threshold
        self.use_vel = use_vel

        print("Use hungarian: {}".format(hungarian))

        if dataset == 'Nuscenes':
            self.velocity_error = NUSCENE_CLS_VELOCITY_ERROR
            self.tracking_names = NUSCENES_TRACKING_NAMES
        elif dataset == 'Waymo':
            self.velocity_error = WAYMO_CLS_VELOCITY_ERROR
            self.tracking_names = WAYMO_TRACKING_NAMES
        self.id_count = 0
        self.tracks = []

        self.reset()

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def det_preprocess(self, results, time_lag):
        temp = []
        for det in results:  # for each detection ...
            # filter out classes not evaluated for tracking
            if det['detection_name'] not in self.tracking_names:
                continue
            # for all evaluated classes, extend with the following attributes
            det['ct'] = np.array(det['translation'][:2])  # ct: 2d centerpoint of one detection
            if self.tracker == 'PointTracker':
                det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
            # label_preds: class id (instead of class name)
            det['label_preds'] = self.tracking_names.index(det['detection_name'])
            temp.append(det)
        return temp

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
                dist[dist > 1e18] = 1e18
                matched_indices = linear_sum_assignment(copy.deepcopy(dist))
            else:
                matched_indices = greedy_assignment(copy.deepcopy(dist))
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

    def split_det(self, det_data, score=0.5):
        '''Split detections by score  
        
        Return `high_score_detections` and `low_score_detections` index
        '''
        scores = np.array([item['detection_score'] for item in det_data], np.float32)
        high_idx = np.where(scores >= score)[0]
        low_idx = np.where(scores < score)[0]
        return high_idx, low_idx

    def split_trk(self, trk_data, times=1):
        '''Split trackers by active or not
        
        Return index of active and non-active
        '''
        active_times = np.array([item['active'] for item in trk_data], np.float32)
        active_idx = np.where(active_times >= times)[0]
        non_active_idx = np.where(active_times < times)[0]
        return active_idx, non_active_idx

    def step_centertrack(self, results, time_lag):
        """
        computes connections between current resources with resources from older frames
        :param results: resources in one specific frame
        :param annotated_data: ground truth for train data
        :param time_lag: time between two successive frame (difference in their timestamp)
        :param version: trainval or test
        :param train_data: boolean true if train_data needed false else
        :param model_path: model_path for learning score update function
        :return: tracks: tracklets (detection + tracking id, age, activity) for one specific frame
                 if train_data true than also return the training data
        """

        if len(results) == 0:   # if no detection in this frame, reset tracks list
            self.tracks = []  # <-- however, this means, all tracklets are gone (i.e. 'died')
            return []
        else:   # if any detection is found, ...
            results = self.det_preprocess(results, time_lag)  # contains all extended resources

        N = len(results)  # number of resources in this frame
        M = len(self.tracks)  # number of tracklets
        ret = []  # initiate return value (will become the updated tracklets list)

        # if no tracklet exist just yet (i.e. processing the first frame)
        if M == 0:
            for result in results:  # for each (extended) detection
                # initiate new tracklet
                track = result
                # initiate threshold (first frame)
                if track['detection_score'] < self.det_th:
                    continue
                self.id_count += 1
                # extend tracklet with the following attributes:
                track['tracking_id'] = self.id_count  # tracklet id
                track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
                track['active'] = self.min_hits  # currently matched? (start with 1)
                # if track['detection_score'] > self.active_th:
                #     track['active'] = self.min_hits
                # else:
                #     track['active'] = 0
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
                ret.append(track)
            self.tracks = ret
            return ret

        # Processing from the second frame
        if self.tracker == 'PointTracker':
            # N X 2
            # dets: estmated 2d centerpoint of a detection in the previous frame (ct + expected offset)
            if 'tracking' in results[0]:
                dets = np.array(
                    [det['ct'].astype(np.float32) + det['tracking'].astype(np.float32)
                     for det in results], np.float32)
            else:
                dets = np.array(
                    [det['ct'] for det in results], np.float32)

            tracks = np.array(
                [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        elif self.tracker == 'KF':
            dets = np.array(
                [det['ct'] for det in results], np.float32)

            tracks = []
            for tracklet in self.tracks:
                tracklet['KF'].predict(F=np.array([[1, 0, time_lag, 0, time_lag * time_lag, 0],
                                                   [0, 1, 0, time_lag, 0, time_lag * time_lag],
                                                   [0, 0, 1, 0, time_lag, 0],
                                                   [0, 0, 0, 1, 0, time_lag],
                                                   [0, 0, 0, 0, 1, 0],
                                                   [0, 0, 0, 0, 0, 1]]))
                tracks.append(tracklet['KF'].x[:2])
            tracks = np.array(tracks, np.float32)  # M x 2

        # Split the detections by score
        h_idx, l_idx = self.split_det(results, score=0.1)
        h_det_data, h_dets = [results[i] for i in h_idx], [dets[i] for i in h_idx]
        l_det_data, l_dets = [results[i] for i in l_idx], [dets[i] for i in l_idx]
        h_dets, l_dets = np.array(h_dets, np.float32), np.array(l_dets, np.float32)
        
        # Split the trackers by active or not
        h_idx, l_idx = self.split_trk(self.tracks, times=1)
        h_track_data, h_tracks = [self.tracks[i] for i in h_idx], [tracks[i] for i in h_idx]
        l_track_data, l_tracks = [self.tracks[i] for i in l_idx], [tracks[i] for i in l_idx]
        h_tracks, l_tracks = np.array(h_tracks, np.float32), np.array(l_tracks, np.float32)

        ''' First association: high score '''
        # matching the current with the estimated pass
        matching = self.comparing_positions(h_track_data, h_det_data, h_tracks, h_dets)
        matched, unmatched_trk, unmatched_det = matching[0], matching[1], matching[2]

        # add matches
        for m in matched:
            # initiate new tracklet (with three additional attributes)
            track = h_det_data[m[0]]
            track['tracking_id'] = h_track_data[m[1]]['tracking_id']  # tracklet id = id of matched trackled
            track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
            track['active'] = h_track_data[m[1]]['active'] + 1
            if self.tracker == 'KF':
                track['KF'] = h_track_data[m[1]]['KF']
                if self.use_vel:
                    track['KF'].update(z=np.hstack([track['ct'], np.array(track['velocity'][:2])]))
                else:
                    track['KF'].update(z=track['ct'])
                track['translation'][0] = track['KF'].x[0]
                track['translation'][1] = track['KF'].x[1]
                track['velocity'][0] = track['KF'].x[2]
                track['velocity'][1] = track['KF'].x[3]
            ret.append(track)

        # unmatch dets use to confirm non-active trackers
        u_det_data_pool = []
        u_dets_pool = []
        for i in unmatched_det:
            u_det_data_pool.append(h_det_data[i])
            u_dets_pool.append(h_dets[i])

        temp_track_data = []
        temp_tracks = []
        for i in unmatched_trk:
            temp_track_data.append(h_track_data[i])
            temp_tracks.append(h_tracks[i])
        h_track_data = temp_track_data
        h_tracks = np.array(temp_tracks, dtype=np.float32)

        ''' Second association: low score '''
        matching = self.comparing_positions(h_track_data, l_det_data, h_tracks, l_dets)
        matched, unmatched_trk, unmatched_det = matching[0], matching[1], matching[2]

        for m in matched:
            # initiate new tracklet (with three additional attributes)
            track = l_det_data[m[0]]
            track['tracking_id'] = h_track_data[m[1]]['tracking_id']  # tracklet id = id of matched trackled
            track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
            track['active'] = h_track_data[m[1]]['active'] + 1
            if self.tracker == 'KF':
                track['KF'] = h_track_data[m[1]]['KF']
                if self.use_vel:
                    track['KF'].update(z=np.hstack([track['ct'], np.array(track['velocity'][:2])]))
                else:
                    track['KF'].update(z=track['ct'])
                track['translation'][0] = track['KF'].x[0]
                track['translation'][1] = track['KF'].x[1]
                track['velocity'][0] = track['KF'].x[2]
                track['velocity'][1] = track['KF'].x[3]
            ret.append(track)

        # Save the unmatch trackers in non-active mode if age < max_age
        for i in unmatched_trk:
            track = h_track_data[i]

            # update score (only apply score decay)
            if self.score_update is not None:
                track['detection_score'] -= self.noise

            # keep tracklet if score is above threshold AND age is not too high
            if track['age'] < self.max_age and track['detection_score'] > self.del_th:
                track['age'] += 1
                # Activate if score is large enough
                if track['detection_score'] > self.s_th:
                    track['active'] += 1
                else:
                    track['active'] = 0

                ct = track['ct']
                if 'tracking' in track:
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                    track['translation'][:2] = track['ct']
                elif 'KF' in track:
                    track['translation'][0] = track['KF'].x[0]
                    track['translation'][1] = track['KF'].x[1]
                    track['velocity'][0] = track['KF'].x[2]
                    track['velocity'][1] = track['KF'].x[3]
                ret.append(track)

        # unmatch dets use to confirm non-active trackers
        for i in unmatched_det:
            u_det_data_pool.append(l_det_data[i])
            u_dets_pool.append(l_dets[i])
        u_dets_pool = np.array(u_dets_pool, dtype=np.float32)

        ''' Deal with unconfirmed tracks: use unmatch dets '''
        matching = self.comparing_positions(l_track_data, u_det_data_pool, l_tracks, u_dets_pool)
        matched, unmatched_trk, unmatched_det = matching[0], matching[1], matching[2]

        for m in matched:
            # initiate new tracklet (with three additional attributes)
            track = u_det_data_pool[m[0]]
            track['tracking_id'] = l_track_data[m[1]]['tracking_id']  # tracklet id = id of matched trackled
            track['age'] = 1  # how many frames without matching detection (i.e. inactivity)
            track['active'] = l_track_data[m[1]]['active'] + 1
            if self.tracker == 'KF':
                track['KF'] = l_track_data[m[1]]['KF']
                if self.use_vel:
                    track['KF'].update(z=np.hstack([track['ct'], np.array(track['velocity'][:2])]))
                else:
                    track['KF'].update(z=track['ct'])
                track['translation'][0] = track['KF'].x[0]
                track['translation'][1] = track['KF'].x[1]
                track['velocity'][0] = track['KF'].x[2]
                track['velocity'][1] = track['KF'].x[3]
            ret.append(track)

        # add unmatched resources as new 'born' tracklets
        for i in unmatched_det:
            track = u_det_data_pool[i]
            if track['detection_score'] < self.det_th:
                continue
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
            if track['detection_score'] > self.det_th:
                track['active'] = 1
            else:
                track['active'] = 0
            ret.append(track)

        # Save the unmatch trackers in non-active mode if age < max_age
        for i in unmatched_trk:
            track = l_track_data[i]

            # update score (only apply score decay)
            if self.score_update is not None:
                track['detection_score'] -= self.noise

            # keep tracklet if score is above threshold AND age is not too high
            if track['age'] < self.max_age and track['detection_score'] > self.del_th:
                track['age'] += 1
                # Activate if score is large enough
                if track['detection_score'] > self.s_th:
                    track['active'] += 1
                else:
                    track['active'] = 0

                ct = track['ct']
                if 'tracking' in track:
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                    track['translation'][:2] = track['ct']
                elif 'KF' in track:
                    track['translation'][0] = track['KF'].x[0]
                    track['translation'][1] = track['KF'].x[1]
                    track['velocity'][0] = track['KF'].x[2]
                    track['velocity'][1] = track['KF'].x[3]
                ret.append(track)

        self.tracks = ret
        return ret