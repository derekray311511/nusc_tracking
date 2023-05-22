# Detection
bbox_score=0.01

# Tracker
tracker='KF'    # KF/PointTracker
use_vel=1
min_hits=1
max_age=6
det_th=0.025
del_th=0.0
active_th=1.0
update_function='multiplication'
score_decay=0.15
use_nms=1
nms_th=0.5

# Radar
radar_fusion=1

# Data and eval
evaluate=0
out_dir="data/track_results/BEVFusion-KF(R)-mul-2-exp-nms_0.5-verror*2"
# out_dir="data/track_results/CenterPoint-KF(R)-mul-1-exp"
dataroot="data/nuscenes"
workspace="/home/Student/Tracking"
split="val"
detection_path="data/track_result_bboxth-0.0/detection_result.json"
# detection_path="data/detection_result.json"   # CenterPoint
frames_meta_path="data/frames_meta.json"
radar_pc_path="data/radar_PC/radar_PC_13Hz_with_vcomp.json"

python tools/track_fusion_exp.py --split $split --workspace $workspace --out-dir $out_dir \
--min_hits $min_hits --det_th $det_th --del_th $del_th --active_th $active_th \
--update_function $update_function --score_decay $score_decay \
--evaluate $evaluate --dataroot $dataroot \
--detection_path $detection_path --frames_meta_path $frames_meta_path \
--tracker $tracker --bbox-score $bbox_score --max_age $max_age \
--use_vel $use_vel --radar_fusion $radar_fusion --radar_pc_path $radar_pc_path \
--use_nms $use_nms --nms_th $nms_th