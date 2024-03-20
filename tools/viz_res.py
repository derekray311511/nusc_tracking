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

from utils.custom_eval import evaluate_nuscenes, draw_boxes

class res_data:
    def __init__(
        self, 
        trk_path_1=None,
        trk_path_2=None,
        radarSeg_path=None,
        radarTrk_path=None,
    ):
        self.trk1 = self.load_trk(trk_path_1)
        self.trk2 = self.load_trk(trk_path_2)
        self.radarSeg = self.load_trk(radarSeg_path)
        self.radarTrk = self.load_trk(radarTrk_path)

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

def analyze_res(
    trk1=None, 
    trk2=None, 
    radarSeg=None, 
    radarTrk=None, 
    gt=None, 
    LiDAR_PC=None,
    image=None,
    ):
    # Show the numbers of targets in each frame with categories
    if trk1 is not None and trk2 is not None:
        print("trk1: ", len(trk1), "trk2: ", len(trk2))
        trk1_categories = [obj['tracking_name'] for obj in trk1]
        trk2_categories = [obj['tracking_name'] for obj in trk2]
        trk1_category_counts = {category: trk1_categories.count(category) for category in set(trk1_categories)}
        trk2_category_counts = {category: trk2_categories.count(category) for category in set(trk2_categories)}
        print("trk1: ", trk1_category_counts); print("trk2: ", trk2_category_counts)
        # Show gt numbers of targets in each frame with categories
        if gt is None:
            pass
        print("gt: ", len(gt))
        gt_categories = [obj['detection_name'] for obj in gt]
        gt_category_counts = {category: gt_categories.count(category) for category in set(gt_categories)}
        print("gt: ", gt_category_counts)

        # Calculate TP, FP, FN in each frame
        TP, FP, FN, matched_predictions = evaluate_nuscenes(trk1, gt)
        print("trk1: ", TP, FP, FN)
        TP, FP, FN, matched_predictions = evaluate_nuscenes(trk2, gt)
        print("trk2: ", TP, FP, FN)


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
        version = "2024-03-18-11:43:1_fuseAllCat_multiRadarIDFix"
    print(f"Visualizing version: {version}...")

    dataset = nusc_dataset(cfg["DATASET"])
    trk_res = res_data(
        trk_path_1=f"/data/early_fusion_track_results/{version}/lidar_tracking_res.json",
        trk_path_2=f"/data/early_fusion_track_results/{version}/tracking_result.json",
        radarSeg_path=f"/data/early_fusion_track_results/{version}/radar_seg_res.json",
        radarTrk_path=f"/data/early_fusion_track_results/{version}/radar_tracking_res.json",
    )
    winName1 = 'LiDAR'
    winName2 = 'Fusion'
    cfg["VISUALIZER"]["trk_res"] = {
        'colorName': True,
        'colorID': False,
        'draw_vel': True,
        'draw_id': True,
        # 'alpha': 0.7, # slow
    }
    trackViz = TrackVisualizer(
        windowName=winName1,
        **cfg["VISUALIZER"],
    )
    trackViz2 = TrackVisualizer(
        windowName=winName2,
        **cfg["VISUALIZER"],
    )
    trackViz3 = TrackVisualizer(
        windowName='radarTrk',
        **cfg["VISUALIZER"],
    )
    winList = [trackViz, trackViz2, trackViz3]
    setWinPos(cfg["VISUALIZER"]["screenSize"], winList)

    frames = dataset.get_frames_meta()
    gts = dataset.get_groundTruth()
    for gt in gts:
        for obj in gt['anns']:
            obj['velocity'] = np.array([0.0, 0.0])
    len_frames = len(frames)
    max_idx = len_frames - 1
    idx = -1
    cv2.createTrackbar('Frame', winName1, 0, max_idx, lambda x: None)
    cv2.createTrackbar('Frame', winName2, 0, max_idx, lambda x: None)
    cv2.setTrackbarPos('Frame', winName1, idx)
    cv2.setTrackbarPos('Frame', winName2, idx)
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
            trackViz.grid = not trackViz.grid
            trackViz2.grid = not trackViz2.grid
        elif key == 13:
            new_idx1 = cv2.getTrackbarPos('Frame', winName1)
            new_idx2 = cv2.getTrackbarPos('Frame', winName2)
            idx = new_idx1 if new_idx1 != idx else new_idx2

        if idx < 0:
            idx = 0
        if idx >= len_frames:
            idx = len_frames - 1

        cv2.setTrackbarPos('Frame', winName1, idx)
        cv2.setTrackbarPos('Frame', winName2, idx)

        token = frames[idx]['token']
        timestamp = frames[idx]['timestamp']
        ego_pose = frames[idx]['ego_pose']
        trk1 = trk_res.trk1[token]
        trk2 = trk_res.trk2[token]
        radarSeg = trk_res.load_radarSeg(token)
        radarTrk = trk_res.radarTrk[token]
        gt = gts[idx]['anns']
        frame_name = "{}-{}".format(timestamp, token)

        trans = dataset.get_4f_transform(ego_pose, inverse=True)
        viz_start = time.time()
        trackViz.draw_ego_car(img_src="/data/car1.png")
        trackViz.draw_det_bboxes(trk1, trans, **cfg["VISUALIZER"]["trk_res"])
        trackViz.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        trackViz.draw_radar_seg(radarSeg, trans, **cfg["VISUALIZER"]["radarSeg"])
        trackViz2.draw_ego_car(img_src="/data/car1.png")
        trackViz2.draw_det_bboxes(trk2, trans, **cfg["VISUALIZER"]["trk_res"])
        trackViz2.draw_det_bboxes(gt, trans, **cfg["VISUALIZER"]["groundTruth"])
        trackViz2.draw_radar_seg(radarSeg, trans, **cfg["VISUALIZER"]["radarSeg"])
        trackViz3.draw_ego_car(img_src="/data/car1.png")
        trackViz3.draw_radar_seg(radarSeg, trans, **cfg["VISUALIZER"]["radarSeg"])
        trackViz3.draw_det_bboxes(radarTrk, trans, **cfg["VISUALIZER"]["radarTrkBox"])
        trackViz.show()
        trackViz2.show()
        trackViz3.show()
        viz_end = time.time()
        if args.show_delay:
            print(f"viz delay:{(viz_end - viz_start) / 1e-3: .2f} ms")

        analyze_res(trk1, trk2, radarSeg, radarTrk, gt, None, 
                    image=np.ones((trackViz.height, trackViz.width, 3), dtype=np.uint8) * trackViz.background_color)
        

if __name__ == "__main__":
    main(get_parser())