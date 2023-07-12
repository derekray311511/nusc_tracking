import argparse
import json
import os
import time
import copy

from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils import splits

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]

def get_parser():
    parser = argparse.ArgumentParser(description="Prepare for tracking")
    parser.add_argument("--work_dir", type=str, default="data/meta_data", help="the dir to save meta data")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    return parser

def name_extraction(names_list, name_to_extract):
    for name in names_list:
        p = name_to_extract.find(name)
        if p != -1:
            np = p + len(name)
            return name_to_extract[p:np]

def save_first_frame(parser):

    args = parser.parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test
    else:
        raise ValueError("unknown")

    frames = []
    # added
    annotations = []
    for sample in tqdm(nusc.sample):
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes:
            continue

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {'token': token, 'timestamp': timestamp}

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True
            sample['first'] = True
        else:
            frame['first'] = False
            sample['first'] = False

        # ego pose
        LIDAR_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', LIDAR_record['ego_pose_token'])
        frame['ego_pose'] = ego_pose

        # added (detection_name is just named for easying the process)
        for annotation in reversed(sample['anns']):
            current = nusc.get("sample_annotation", annotation)
            current['detection_name'] = name_extraction(NUSCENES_TRACKING_NAMES, current['category_name'])

            if current['detection_name'] in NUSCENES_TRACKING_NAMES:
                current['label_preds'] = int(NUSCENES_TRACKING_NAMES.index(current['detection_name']))
                sample['anns'][sample['anns'].index(annotation)] = current
            else:
                sample['anns'].remove(annotation)

        sample_filtered = sample.copy()
        del sample_filtered['data']

        frames.append(frame)
        annotations.append(sample_filtered)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)
    # added
    with open(os.path.join(args.work_dir, 'annotations.json'), "w") as f:
        json.dump({'samples': annotations}, f)

if __name__ == "__main__":
    save_first_frame(get_parser())