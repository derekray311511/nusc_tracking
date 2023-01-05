detection_path="/data/track_result_bboxth-0.0/detection_result.json"
tracking_path="/data/track_results/BEVFusion-radar-KF-fuse_new/tracking_result.json"
# detection_path="/data/detection_result.json"
# tracking_path="/data/track_results/CenterPoint-radar-KF-fuse(new)/tracking_result.json"
radar_trk_path="/data/radar_PC/radar_tracking_result_13Hz.json"
viz_radar_trks=1

vis_th=0.05
vis_img_bbox=1

lidar_stack=1
radar_stack=1
det_bbox_stack=1
trk_bbox_stack=5
id_color=1
init_idx=450

multi_thread=1

python visualize_opt.py --detection_path $detection_path --track_res_path $tracking_path \
--vis_th $vis_th --init_idx $init_idx --vis_img_bbox $vis_img_bbox \
--lidar_stack $lidar_stack --radar_stack $radar_stack \
--det_bbox_stack $det_bbox_stack --trk_bbox_stack $trk_bbox_stack \
--multi_thread $multi_thread --radar_trk_path $radar_trk_path \
--viz_radar_trks $viz_radar_trks --id_color $id_color
