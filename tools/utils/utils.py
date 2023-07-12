from collections import deque
from datetime import datetime
import time
import os, sys
import pytz
import json

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