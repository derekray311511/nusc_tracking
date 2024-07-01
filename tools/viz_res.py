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
from utils.box_utils import get_3d_box_8corner, get_3d_box_2corner
from utils.box_utils import nms, is_points_inside_obb
from utils.geometry_utils import *
from utils.visualizer import TrackVisualizer
from utils.nusc_eval import nusc_eval
from utils.nusc_util import nusc_dataset
from tracker import PubTracker as lidarTracker
from early_fusion_tracker import PubTracker as radarTracker
from early_fusion_fusion import Fusion

from utils.custom_eval import evaluate_nuscenes, TrackingEvaluation

class res_data:
    def __init__(
        self, 
        det_path=None,
        trk_path_1=None,
        trk_path_2=None,
        radarSeg_path=None,
        radarTrk_path=None,
    ):
        self.det = self.load_det(det_path)
        self.trk1 = self.load_trk(trk_path_1)
        self.trk2 = self.load_trk(trk_path_2)
        self.radarSeg = self.load_trk(radarSeg_path)
        self.radarTrk = self.load_trk(radarTrk_path)

    def load_det(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)['results']
        return data

    def load_trk(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)['results']
        return data

    def load_radarSeg(self, token):
        seg_f = self.radarSeg[token]
        seg = np.zeros((len(seg_f), 7))
        for i in range(len(seg_f)):
            seg[i][:3] = seg_f[i]['translation']
            seg[i][3:5] = seg_f[i]['velocity']
            seg[i][5] = seg_f[i]['segment_id']
            seg[i][6] = seg_f[i]['category']
        return seg

    def get_det_bbox(self, token, threshold=0.0):
        det = self.det[token]
        det_bbox = []
        for obj in det:
            if obj['detection_score'] > threshold:
                det_bbox.append(obj)
        return det_bbox

def setWinPos(screenSize, winList):
    screen_width, screen_height = screenSize  # Replace with your screen resolution
    window_width = screen_width // 2  # Half of the screen width
    window_height = screen_height  # Full height
    winNum = len(winList)
    # Two main windows and other small windows
    if winNum < 2:
        cv2.moveWindow(winList[0].windowName, 0, 0)
    else:
        cv2.moveWindow(winList[0].windowName, 0, 0)
        cv2.moveWindow(winList[1].windowName, window_width, 0)
    
    if winNum > 2:
        for i in range(2, winNum, 1):
            win = winList[i]
            w, h = win.windowSize
            cv2.moveWindow(win.windowName, window_width - int(w/2), (i-2) * h)

class TrackEval(object):
    def __init__(self):
        self.trackEval_1 = TrackingEvaluation()
        self.trackEval_2 = TrackingEvaluation()
        self.total_data = {'trk1': [0, 0, 0], 'trk2': [0, 0, 0]}
        self.score_history = {'trk1': {}, 'trk2': {}}   # id : score list
        self.duplicate_num = 0

    def record_score(self, trk, score_history):
        for obj in trk:
            tracking_id = int(float(obj['tracking_id']))
            if tracking_id not in score_history:
                score_history[tracking_id] = []
            score_history[tracking_id].append(obj['tracking_score'])

    def get_average_score(self, score_history):
        avg_score = {}
        for id, scores in score_history.items():
            avg_score[id] = sum(scores) / len(scores)
        return avg_score

    def get_matched_objects(self, matched_ids, objects):
        """
        Extract objects matching the provided IDs from a list of objects.
        """
        if len(objects) == 0:
            return []
        if 'tracking_id' in objects[0]:
            return [obj for obj in objects if obj['tracking_id'] in matched_ids]
        else:
            return [obj for obj in objects if obj['instance_token'] in matched_ids]

    def analyze_res(self, trk1=None, trk2=None, gt=None, distance_threshold=2.0, accumulate=True):
        # Show the numbers of targets in each frame with categories
        if trk1 is not None and trk2 is not None:
            trk1_categories = [obj['tracking_name'] for obj in trk1]
            trk2_categories = [obj['tracking_name'] for obj in trk2]
            trk1_category_counts = {category: trk1_categories.count(category) for category in set(trk1_categories)}
            trk2_category_counts = {category: trk2_categories.count(category) for category in set(trk2_categories)}

            for obj in trk1:
                obj['tracking_id'] = int(float(obj['tracking_id']))

            id_list = [obj['tracking_id'] for obj in trk2]
            if len(id_list) != len(set(id_list)):
                self.duplicate_num += len(id_list) - len(set(id_list))
                print("Duplicate tracking IDs found in trk2.")
            else:
                print("No duplicate tracking IDs found in trk2.")

            print("trk1 len:", len(trk1), ", trk2 len:", len(trk2))
            print("trk1:", trk1_category_counts)
            print("trk2:", trk2_category_counts)

            # Show gt numbers of targets in each frame with categories
            if gt is None:
                pass

            gt = deepcopy(gt)
            gt_categories = [obj['detection_name'] for obj in gt]
            gt_category_counts = {category: gt_categories.count(category) for category in set(gt_categories)}

            print("gt:", len(gt))
            print("gt:", gt_category_counts)

            matched_pred_trk1, matched_gt_trk1 = [], []
            matched_pred_trk2, matched_gt_trk2 = [], []
            categories = set(trk1_categories + trk2_categories)
            for category in categories:
                # Filter predictions and ground truths for the current category
                trk1_cat = [obj for obj in trk1 if obj['tracking_name'] == category]
                trk2_cat = [obj for obj in trk2 if obj['tracking_name'] == category]
                gt_cat = [obj for obj in gt if obj['detection_name'] == category]

                # Calculate TP, FP, FN for the current category
                mota_1_cat, _ = self.trackEval_1.evaluate_nuscenes_mota(trk1_cat, gt_cat, distance_threshold=distance_threshold)
                mota_2_cat, _ = self.trackEval_2.evaluate_nuscenes_mota(trk2_cat, gt_cat, distance_threshold=distance_threshold)

                matched_pred_ids_trk1_cat = mota_1_cat.events[mota_1_cat.events['Type'] == 'MATCH']['HId'].unique()
                matched_gt_ids_trk1_cat = mota_1_cat.events[mota_1_cat.events['Type'] == 'MATCH']['OId'].unique()
                matched_pred_trk1 += self.get_matched_objects(matched_pred_ids_trk1_cat, trk1_cat)
                matched_gt_trk1 += self.get_matched_objects(matched_gt_ids_trk1_cat, gt_cat)
                
                matched_pred_ids_trk2_cat = mota_2_cat.events[mota_2_cat.events['Type'] == 'MATCH']['HId'].unique()
                matched_gt_ids_trk2_cat = mota_2_cat.events[mota_2_cat.events['Type'] == 'MATCH']['OId'].unique()
                matched_pred_trk2 += self.get_matched_objects(matched_pred_ids_trk2_cat, trk2_cat)
                matched_gt_trk2 += self.get_matched_objects(matched_gt_ids_trk2_cat, gt_cat)
                
                tp_trk1 = len(matched_gt_ids_trk1_cat)
                fp_trk1 = len([obj for obj in trk1_cat if obj['tracking_id'] not in matched_pred_ids_trk1_cat])
                fn_trk1 = len([obj for obj in gt_cat if obj['instance_token'] not in matched_gt_ids_trk1_cat])

                tp_trk2 = len(matched_gt_ids_trk2_cat)
                fp_trk2 = len([obj for obj in trk2_cat if obj['tracking_id'] not in matched_pred_ids_trk2_cat])
                fn_trk2 = len([obj for obj in gt_cat if obj['instance_token'] not in matched_gt_ids_trk2_cat])

                print(f"Category: {category}")
                print(f"Tracker 1 - TP: {tp_trk1}, FP: {fp_trk1}, FN: {fn_trk1}")
                print(f"Tracker 2 - TP: {tp_trk2}, FP: {fp_trk2}, FN: {fn_trk2}")

                # If accumulate is True, add to total data
                if accumulate:
                    self.total_data['trk1'] = [x + y for x, y in zip(self.total_data['trk1'], [tp_trk1, fp_trk1, fn_trk1])]
                    self.total_data['trk2'] = [x + y for x, y in zip(self.total_data['trk2'], [tp_trk2, fp_trk2, fn_trk2])]

                mota_1_cat.reset()
                mota_2_cat.reset()

            # # Calculate TP, FP, FN for the current category
            # mota_1, _ = self.trackEval_1.evaluate_nuscenes_mota(trk1, gt, distance_threshold=distance_threshold)
            # mota_2, _ = self.trackEval_2.evaluate_nuscenes_mota(trk2, gt, distance_threshold=distance_threshold)

            # matched_pred_ids_trk1 = mota_1.events[mota_1.events['Type'] == 'MATCH']['HId'].unique()
            # matched_gt_ids_trk1 = mota_1.events[mota_1.events['Type'] == 'MATCH']['OId'].unique()
            # matched_pred_trk1 += self.get_matched_objects(matched_pred_ids_trk1, trk1)
            # matched_gt_trk1 += self.get_matched_objects(matched_gt_ids_trk1, gt)
            
            # matched_pred_ids_trk2 = mota_2.events[mota_2.events['Type'] == 'MATCH']['HId'].unique()
            # matched_gt_ids_trk2 = mota_2.events[mota_2.events['Type'] == 'MATCH']['OId'].unique()
            # matched_pred_trk2 += self.get_matched_objects(matched_pred_ids_trk2, trk2)
            # matched_gt_trk2 += self.get_matched_objects(matched_gt_ids_trk2, gt)
            
            # tp_trk1 = len(matched_gt_ids_trk1)
            # fp_trk1 = len([obj for obj in trk1 if obj['tracking_id'] not in matched_pred_ids_trk1])
            # fn_trk1 = len([obj for obj in gt if obj['instance_token'] not in matched_gt_ids_trk1])

            # tp_trk2 = len(matched_gt_ids_trk2)
            # fp_trk2 = len([obj for obj in trk2 if obj['tracking_id'] not in matched_pred_ids_trk2])
            # fn_trk2 = len([obj for obj in gt if obj['instance_token'] not in matched_gt_ids_trk2])

            # print(f"Tracker 1 - TP: {tp_trk1}, FP: {fp_trk1}, FN: {fn_trk1}")
            # print(f"Tracker 2 - TP: {tp_trk2}, FP: {fp_trk2}, FN: {fn_trk2}")

            # # If accumulate is True, add to total data
            # if accumulate:
            #     self.total_data['trk1'] = [x + y for x, y in zip(self.total_data['trk1'], [tp_trk1, fp_trk1, fn_trk1])]
            #     self.total_data['trk2'] = [x + y for x, y in zip(self.total_data['trk2'], [tp_trk2, fp_trk2, fn_trk2])]

            # mota_1.reset()
            # mota_2.reset()

            print("total trk1 TP, FP, FN:", self.total_data['trk1'])
            print("total trk2 TP, FP, FN:", self.total_data['trk2'])
            print(f"Duplicate objects number: {self.duplicate_num}")

            # self.record_score(trk1, self.score_history['trk1'])
            # self.record_score(trk2, self.score_history['trk2'])
            # avg_score_trk1 = self.get_average_score(self.score_history['trk1'])
            # avg_score_trk2 = self.get_average_score(self.score_history['trk2'])
            # sorted_avg_score_trk1 = dict(sorted(avg_score_trk1.items(), key=lambda x: x[0]))
            # sorted_avg_score_trk2 = dict(sorted(avg_score_trk2.items(), key=lambda x: x[0]))
            # print("Score trk1:")
            # for key, value in sorted_avg_score_trk1.items():
            #     print(f"{key}: {value}")
            # print("Score trk2:")
            # for key, value in sorted_avg_score_trk2.items():
            #     print(f"{key}: {value}")

        return (matched_pred_trk1, matched_gt_trk1), (matched_pred_trk2, matched_gt_trk2)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LRFusion visualize")
    parser.add_argument("config", metavar="CONFIG_FILE")
    parser.add_argument("--workspace", type=str, default="/home/Student/Tracking")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--show_delay", action="store_true")
    return parser

def main(parser) -> None:
    args, opts = parser.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    print("cv2 version: ", cv2.__version__)

    if args.version is not None:
        version = args.version
    else:
        version = "2024-05-30-15:34:18_noFusePedTruck(Good)"
    print(f"Visualizing version: {version}...")

    dataset = nusc_dataset(cfg["DATASET"])
    trk_res = res_data(
        det_path="/data/track_result_bboxth-0.0/detection_result.json",
        trk_path_1=f"/data/early_fusion_track_results/{version}/lidar_tracking_res.json",
        trk_path_2=f"/data/early_fusion_track_results/{version}/tracking_result.json",
        radarSeg_path=f"/data/early_fusion_track_results/{version}/radar_seg_res.json",
        radarTrk_path=f"/data/early_fusion_track_results/{version}/radar_tracking_res.json",
    )
    trackEval = TrackEval()
    winName1 = 'LiDAR'
    winName2 = 'Fusion'
    cfg["VISUALIZER"]["trk_res"] = {
        'colorName': True,
        'colorID': False,
        'draw_vel': True,
        'draw_id': False,
        'draw_name': False,
        'draw_score': False,
        'draw_hist': False, # Could cause slow visualization
        'legend': True, 
        # 'alpha': 0.7, # slow
    }
    cfg["VISUALIZER"]["analyze"] = {
        'draw_id': False,
        'draw_name': False,
        'draw_score': False,
        'draw_vel': True, 
        'legend': True, 
    }
    cfg["VISUALIZER"]["detBox"] = {
        'draw': False, 
        'draw_id': False,
        'draw_name': False,
        'draw_score': False,
        'legend': False,
        **cfg["VISUALIZER"]["detBox"],
    }
    cfg["VISUALIZER"]["radarTrkBox"]["draw_id"] = cfg["VISUALIZER"]["trk_res"]["draw_id"]
    cfg["VISUALIZER"]["radarTrkBox"]["draw_name"] = cfg["VISUALIZER"]["trk_res"]["draw_name"]
    cfg["VISUALIZER"]["radarTrkBox"]["draw_score"] = cfg["VISUALIZER"]["trk_res"]["draw_score"]
    trackViz = TrackVisualizer(
        windowName=winName1,
        **cfg["VISUALIZER"],
    )
    trackViz2 = TrackVisualizer(
        windowName=winName2,
        **cfg["VISUALIZER"],
    )
    # trackViz3 = TrackVisualizer(
    #     windowName='radarTrk',
    #     **cfg["VISUALIZER"],
    # )
    analyzeWin1 = TrackVisualizer(
        windowName='analyzeTrk1',
        **cfg["VISUALIZER"],
    )
    analyzeWin2 = TrackVisualizer(
        windowName='analyzeTrk2',
        **cfg["VISUALIZER"],
    )
    winList = [trackViz, trackViz2, analyzeWin1, analyzeWin2]
    setWinPos(cfg["VISUALIZER"]["screenSize"], winList)

    frames = dataset.get_frames_meta()
    gts = dataset.get_groundTruth()
    idx_record = set()
    for gt in gts:
        for obj in gt['anns']:
            obj['velocity'] = np.array([0.0, 0.0])
    len_frames = len(frames)
    current_first_frame_idx = 0
    max_idx = len_frames - 1
    idx = -1
    for win in winList:
        cv2.createTrackbar('Frame', win.windowName, 0, max_idx, lambda x: None)
        cv2.setTrackbarPos('Frame', win.windowName, idx)
        
    while True:

        if trackViz.play:
            idx += 1
            key = cv2.waitKey(int(trackViz.duration * 1000))
        else:
            key = cv2.waitKey(0)

        if key == 27: # esc
            cv2.destroyAllWindows()
            exit(0)
        elif key == 100 or key == 83 or key == 54: # d
            idx += 1
        elif key == 97 or key == 81 or key == 52: # a
            idx -= 1
        elif key == 32: # space
            trackViz.play = not trackViz.play
        elif key == 43: # +
            trackViz.duration *= 2
            print(f"Viz duration set to {trackViz.duration}")
        elif key == 45: # -
            trackViz.duration *= 0.5
            print(f"Viz duration set to {trackViz.duration}")
        elif key == ord('g'):
            for win in winList:
                win.grid = not win.grid
        elif key == ord('i'):
            cfg["VISUALIZER"]["trk_res"]["draw_id"] = not cfg["VISUALIZER"]["trk_res"]["draw_id"]
            cfg["VISUALIZER"]["analyze"]["draw_id"] = not cfg["VISUALIZER"]["analyze"]["draw_id"]
            cfg["VISUALIZER"]["radarTrkBox"]["draw_id"] = not cfg["VISUALIZER"]["radarTrkBox"]["draw_id"]
        elif key == ord('c'):
            cfg["VISUALIZER"]["groundTruth"]["colorName"] = not cfg["VISUALIZER"]["groundTruth"]["colorName"]
        elif key == ord('n'):
            cfg["VISUALIZER"]["trk_res"]["draw_name"] = not cfg["VISUALIZER"]["trk_res"]["draw_name"]
            cfg["VISUALIZER"]["analyze"]["draw_name"] = not cfg["VISUALIZER"]["analyze"]["draw_name"]
            cfg["VISUALIZER"]["radarTrkBox"]["draw_name"] = not cfg["VISUALIZER"]["radarTrkBox"]["draw_name"]
            cfg["VISUALIZER"]["detBox"]["draw_name"] = cfg["VISUALIZER"]["trk_res"]["draw_name"]
        elif key == ord('s'):
            cfg["VISUALIZER"]["trk_res"]["draw_score"] = not cfg["VISUALIZER"]["trk_res"]["draw_score"]
            cfg["VISUALIZER"]["analyze"]["draw_score"] = not cfg["VISUALIZER"]["analyze"]["draw_score"]
            cfg["VISUALIZER"]["radarTrkBox"]["draw_score"] = not cfg["VISUALIZER"]["radarTrkBox"]["draw_score"]
            cfg["VISUALIZER"]["detBox"]["draw_score"] = cfg["VISUALIZER"]["trk_res"]["draw_score"]
        elif key == ord('1'):
            cfg["VISUALIZER"]["detBox"]["draw"] = not cfg["VISUALIZER"]["detBox"]["draw"]
        elif key == ord('h'):
            cfg["VISUALIZER"]["trk_res"]["draw_hist"] = not cfg["VISUALIZER"]["trk_res"]["draw_hist"]
        elif key == 13: # enter
            for win in winList:
                winName = win.windowName
                det_idx = cv2.getTrackbarPos('Frame', winName)
                if det_idx != idx:
                    idx = det_idx
                    break
                    

        if idx < 0:
            idx = 0
        # if idx >= 1000:
        #     idx = 1000 - 1

        if frames[idx]['first']:
            current_first_frame_idx = idx

        for win in winList:
            cv2.setTrackbarPos('Frame', win.windowName, idx)

        token = frames[idx]['token']
        timestamp = frames[idx]['timestamp']
        ego_pose = frames[idx]['ego_pose']
        det = trk_res.get_det_bbox(token, threshold=cfg["VISUALIZER"]["detBox"]["score_th"])
        trk1 = trk_res.trk1[token]
        trk2 = trk_res.trk2[token]
        radarSeg = trk_res.load_radarSeg(token)
        radarTrk = trk_res.radarTrk[token]
        gt = gts[idx]['anns']
        frame_name = "{}-{}".format(timestamp, token)

        if idx not in idx_record:
            accumulate = True
        else:
            accumulate = False
        idx_record.add(idx)
        if frames[idx]['first']:
            trackEval.score_history = {'trk1': {}, 'trk2': {}}
        eval_trk1, eval_trk2 = trackEval.analyze_res(trk1, trk2, gt, distance_threshold=3.0, accumulate=accumulate)

        trans = dataset.get_4f_transform(ego_pose, inverse=True)
        viz_start = time.time()
        if cfg["VISUALIZER"]["trk_res"]["draw_hist"]:
            hist_num = 4
            for i in reversed(range(hist_num)):
                if idx - i - 1 == current_first_frame_idx:
                    continue
                token = frames[idx - i - 1]['token']
                alpha = 1.0 - 0.9 * (i + 1) / 5
                trackViz.draw_det_bboxes(trk_res.trk1[token], trans, **cfg["VISUALIZER"]["trk_res"], alpha=alpha)
                trackViz2.draw_det_bboxes(trk_res.trk2[token], trans, **cfg["VISUALIZER"]["trk_res"], alpha=alpha)
        trackViz.draw_ego_car(img_src="/data/car1.png")
        trackViz.draw_det_bboxes(trk1, trans, **cfg["VISUALIZER"]["trk_res"])
        trackViz.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        trackViz.draw_radar_seg(radarSeg, trans, **cfg["VISUALIZER"]["radarSeg"])
        trackViz2.draw_ego_car(img_src="/data/car1.png")
        trackViz2.draw_det_bboxes(trk2, trans, **cfg["VISUALIZER"]["trk_res"])
        trackViz2.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        trackViz2.draw_radar_seg(radarSeg, trans, **cfg["VISUALIZER"]["radarSeg"])
        # trackViz3.draw_ego_car(img_src="/data/car1.png")
        # trackViz3.draw_radar_seg(radarSeg, trans, **cfg["VISUALIZER"]["radarSeg"])
        # trackViz3.draw_det_bboxes(radarTrk, trans, **cfg["VISUALIZER"]["radarTrkBox"])
        # trackViz3.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        analyzeWin1.draw_ego_car(img_src="/data/car1.png")
        analyzeWin1.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        analyzeWin1.drawTP_FP_FN(trk1, gt, eval_trk1[0], eval_trk1[1], trans, **cfg["VISUALIZER"]["analyze"])
        analyzeWin2.draw_ego_car(img_src="/data/car1.png")
        analyzeWin2.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        analyzeWin2.drawTP_FP_FN(trk2, gt, eval_trk2[0], eval_trk2[1], trans, **cfg["VISUALIZER"]["analyze"])
        for win in winList:
            if cfg["VISUALIZER"]["detBox"]["draw"]:
                win.draw_det_bboxes(det, trans, **cfg["VISUALIZER"]["detBox"])
            win.show()

        viz_end = time.time()
        if args.show_delay:
            print(f"viz delay:{(viz_end - viz_start) / 1e-3: .2f} ms")

        print()
        

if __name__ == "__main__":
    main(get_parser())