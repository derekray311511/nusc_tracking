import numpy as np
import cv2
import sklearn
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom

def calculate_distance(gt_translation, pred_translation):
    return np.linalg.norm(np.array(gt_translation) - np.array(pred_translation))

def evaluate_nuscenes(predictions, ground_truths, distance_threshold=2.0):
    TP = 0
    FP = 0
    FN = 0
    matched_predictions = []
    matched_gt = []

    for gt in ground_truths:
        matched = False
        for pred in predictions:
            distance = calculate_distance(gt['translation'], pred['translation'])
            if distance < distance_threshold:
                matched = True
                if pred not in matched_predictions:
                    TP += 1
                    matched_gt.append(gt)
                    matched_predictions.append(pred)
                break
        if not matched:
            FN += 1

    FP = len(predictions) - len(matched_predictions)
    return TP, FP, FN, matched_predictions, matched_gt

# Assume `predictions` and `ground_truths` are lists of dictionaries with 'translation', 'size', 'velocity'


class TrackingEvaluation(object):
    def __init__(self):
        self.motaAccumulator = MOTAccumulatorCustom()   # one for each scene
        self.frame_id = 0

    def evaluate_nuscenes_mota(self, frame_pred, frame_gt, distance_threshold=2.0):
        """
        Evaluate the tracking results using the MOTA metric.
        """
        mota = self.motaAccumulator
        
        # Abort if there are neither GT nor pred boxes.
        gt_ids = [gg['instance_token'] for gg in frame_gt]
        pred_ids = [tt['tracking_id'] for tt in frame_pred]
        if len(gt_ids) == 0 and len(pred_ids) == 0:
            return self.motaAccumulator, []

        if len(frame_gt) == 0 or len(frame_pred) == 0:
            distances = np.ones((0, 0))
        else:
            gt_boxes = np.array([b['translation'][:2] for b in frame_gt])
            pred_boxes = np.array([b['translation'][:2] for b in frame_pred])
            distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

        # Distances that are larger than the threshold won't be associated.
        assert len(distances) == 0 or not np.all(np.isnan(distances))
        distances[distances >= distance_threshold] = np.nan

        # Accumulate results.
        # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
        mota.update(gt_ids, pred_ids, distances, frameid=self.frame_id)

        events = mota.events.loc[self.frame_id]
        matches = events[events.Type == 'MATCH']
        match_ids = matches.HId.values
        match_scores = [tt['tracking_score'] for tt in frame_pred if tt['tracking_id'] in match_ids]

        self.frame_id += 1

        return mota, match_scores