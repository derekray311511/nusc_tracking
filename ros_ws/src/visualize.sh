detection_path="/data/detection_result.json"
tracking_path="/data/track_results/tracking_result.json"

vis_th=0.05
vis_img_bbox=1

lidar_stack=1
radar_stack=1
det_bbox_stack=1
trk_bbox_stack=5
init_idx=0

python visualize.py --detection_path $detection_path --track_res_path $tracking_path \
--vis_th $vis_th --init_idx $init_idx --vis_img_bbox $vis_img_bbox \
--lidar_stack $lidar_stack --radar_stack $radar_stack \
--det_bbox_stack $det_bbox_stack --trk_bbox_stack $trk_bbox_stack
