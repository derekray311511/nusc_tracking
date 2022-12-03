split="val"
out_dir="data/track_results/CenterPoint"
min_hits=1
det_th=0
evaluate=1
dataroot="data/nuscenes"
detection_path="data/detection_result.json"
frames_meta_path="data/frames_meta.json"
bbox_score=0.0

python tools/track.py --split $split --out-dir $out_dir --min_hits $min_hits \
--det_th $det_th --evaluate $evaluate --dataroot $dataroot \
--detection_path $detection_path --frames_meta_path $frames_meta_path \
--bbox-score $bbox_score
