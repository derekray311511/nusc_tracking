# Result path
dataroot="/data/small_data2"
datetime="2023-01-11-19:18:23"
radar_trk_path="/data/radar_PC/$datetime/radar_tracking_result_13Hz.json"
detection_path="/data/track_result_bboxth-0.0/detection_result.json"
tracking1_path="/data/track_results/BEVFusion/tracking_result.json"
tracking2_path="/data/track_results/BEVFusion($datetime)/tracking_result.json"
# detection_path="/data/detection_result.json"
# tracking1_path="/data/track_results/CenterPoint/tracking_result.json"
# tracking2_path="/data/track_results/CenterPoint($datetime)/tracking_result.json"

vis_th=0.1
vis_img_bbox=1
viz_radar_trks=1
viz_r_vel=1

lidar_stack=1
radar_stack=1
det_bbox_stack=1
trk_bbox_stack=5
id_color=0

multi_thread=1
init_idx=1000

python visualize_opt.py --detection_path $detection_path --track1_res_path $tracking1_path --track2_res_path $tracking2_path \
--vis_th $vis_th --init_idx $init_idx --vis_img_bbox $vis_img_bbox \
--lidar_stack $lidar_stack --radar_stack $radar_stack \
--det_bbox_stack $det_bbox_stack --trk_bbox_stack $trk_bbox_stack \
--multi_thread $multi_thread --radar_trk_path $radar_trk_path \
--viz_radar_trks $viz_radar_trks --id_color $id_color \
--dataroot $dataroot --viz_r_vel $viz_r_vel
