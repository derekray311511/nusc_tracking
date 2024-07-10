from collections import deque
from datetime import datetime
from typing import Dict, Tuple
import time
import os, sys
import pytz
import json
import numpy as np

def cal_func_time(func, **kargs):
    start = time.time()
    ret = func(**kargs)
    end = time.time()
    return ret, end - start

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def get_current_datetime(zone='Asia/Taipei'):
    time = datetime.now(pytz.timezone(zone))
    time = f"{time.year:04d}-{time.month:02d}-{time.day:02d}-{time.hour:02d}:{time.minute:02d}:{time.second}"
    return time

def log_parser_args(out_dir, args):
    log_path = os.path.join(out_dir, 'log.json')
    info = {}
    info.update(vars(args))
    mkdir_or_exist(os.path.dirname(log_path))
    with open(log_path, 'w') as f:
        json.dump(info, f, indent=2)
    f.close()
    print(f"Log file is saved to {log_path}")

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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
    def __init__(self, tracker) -> None:
        self.centers = {}
        self.vels = {}
        self.trackers = tracker

    def save_path(self, center_list):
        for obj in center_list:
            id = obj['tracking_id']
            self.centers[id] = obj['translation']
            self.vels[id] = obj['velocity']

        for track_id in self.centers:
            if track_id in self.trackers:  
                self.trackers[track_id].update(self.centers[track_id], self.vels[track_id])
            else:   
                self.trackers[track_id] = Object(self.centers[track_id], self.vels[track_id])

        for track_id in self.trackers:    
            if track_id not in self.centers:
                self.trackers[track_id].update(None, None)

        return self.trackers

def get_trk_colormap() -> Dict[str, Tuple[int, int, int]]:
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective BGR values.
    """

    classname_to_color = {  # BGR.
        "pedestrian": (230, 0, 0),  # Blue
        "bicycle": (60, 20, 220),  # Crimson
        "bus": (80, 127, 255),  # Coral
        "car": (0, 158, 255),  # Orange
        "motorcycle": (99, 61, 255),  # Red
        "trailer": (0, 140, 255),  # Darkorange
        "truck": (71, 99, 255),  # Tomato
        "background": (0, 0, 255),  # For no segmented radar object
    }

    return classname_to_color

def encodeCategory(cats, categories):
    """ Encode categories to numbers 
    Param:
        cats: list of categories (str)
    Return:
        ret: list of int
    """
    ret = []
    for cat in cats:
        num = categories.index(cat) if cat in categories else -1
        ret.append(num)
    return ret

def decodeCategory(nums, categories):
    """ Encode categories to numbers 
    Param:
        nums: list of int
    Return:
        ret: list of categories (str)
    """
    ret = []
    for num in nums:
        cat = categories[num] if num < len(categories) else None
        ret.append(cat)
    return ret