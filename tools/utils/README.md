# Customize tracking evaluation

- Change this file: /opt/conda/lib/python3.8/site-packages/nuscenes/eval/tracking/loaders.py
- Change this function: create_tracks

```python
import json
def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes, eval_split: str, gt: bool) \
        -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
    This can be applied either to GT or predictions.
    :param all_boxes: Holds all GT or predicted boxes.
    :param nusc: The NuScenes instance to load the sample information from.
    :param eval_split: The evaluation split for which we create tracks.
    :param gt: Whether we are creating tracks for GT or predictions
    :return: The tracks.
    """
    # Custom eval tokens
    frames_path = "/data/meta_data2/frames_meta.json"
    with open(frames_path, 'rb') as f:
        frames = json.load(f)['frames']
    custom_tokens = []
    use_custom = True
    if use_custom:
        for idx in range(1000):
            custom_tokens.append(frames[idx]['token'])
    else:
        for sample_token in all_boxes.sample_tokens:
            custom_tokens.append(sample_token)
    print(f"Evaluation using {len(custom_tokens)} frames.")

    # Only keep samples from this split.
    splits = create_splits_scenes()
    scene_tokens = set()
    # for sample_token in all_boxes.sample_tokens:  # all eval tokens
    for sample_token in custom_tokens:  # custom
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene = nusc.get('scene', scene_token)
        if scene['name'] in splits[eval_split]:
            scene_tokens.add(scene_token)

    # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
    tracks = defaultdict(lambda: defaultdict(list))

    # Init all scenes and timestamps to guarantee completeness.
    for scene_token in scene_tokens:
        # Init all timestamps in this scene.
        scene = nusc.get('scene', scene_token)
        cur_sample_token = scene['first_sample_token']
        while True:
            # Initialize array for current timestamp.
            cur_sample = nusc.get('sample', cur_sample_token)
            tracks[scene_token][cur_sample['timestamp']] = []

            # Abort after the last sample.
            if cur_sample_token == scene['last_sample_token']:
                break

            # Move to next sample.
            cur_sample_token = cur_sample['next']

    # Group annotations wrt scene and timestamp.
    # for sample_token in all_boxes.sample_tokens:  # all eval tokens
    for sample_token in custom_tokens:  # custom
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        tracks[scene_token][sample_record['timestamp']] = all_boxes.boxes[sample_token]

    # Replace box scores with track score (average box score). This only affects the compute_thresholds method and
    # should be done before interpolation to avoid diluting the original scores with interpolated boxes.
    if not gt:
        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # Compute average scores for each track.
            track_id_avg_scores = {}
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[tracking_id] = np.mean(scores)

            # Apply average score to each box.
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = track_id_avg_scores[box.tracking_id]

    # Interpolate GT and predicted tracks.
    for scene_token in tracks.keys():
        tracks[scene_token] = interpolate_tracks(tracks[scene_token])

        if not gt:
            # Make sure predictions are sorted in in time. (Always true for GT).
            tracks[scene_token] = defaultdict(list, sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))

    return tracks
```