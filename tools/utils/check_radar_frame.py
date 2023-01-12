import json
import os
import numpy as np

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def load_frames_meta(path):
    with open(path, 'rb') as f:
        frames = json.load(f)
    return frames['frames']

def load_radar_trk(path):
    if path is None:
        return None
    with open(path, 'rb') as f:
        data = json.load(f)
        try:
            radar_trks = data['key_results']
        except: # Old version
            radar_trks = data['results']
    return radar_trks

def logging(root, file_name, infos):
    mkdir_or_exist(root)
    with open(os.path.join(root, file_name), 'w') as f:
        for key, value in infos.items(): 
            f.write('%s:%s\n' % (key, value))
    f.close()
    return

def main():
    root = '/home/Student/Tracking'
    frame_data_path = 'data/frames_meta.json'
    radar_data_path = 'data/radar_PC/radar_tracking_result_13Hz.json'

    frames = load_frames_meta(frame_data_path)
    radar_trks = load_radar_trk(radar_data_path)

    # Record radar no frame infos
    infos = {}

    length = len(frames)
    for i in range(length):
        token = frames[i]['token']
        try:
            _ = radar_trks[token]
        except:
            infos.update({i: token})
    
    logging(os.path.join(root, 'debug'), 'radar_miss_tokens_new.txt', infos)

if __name__ == "__main__":
    main()
    print("Done")
