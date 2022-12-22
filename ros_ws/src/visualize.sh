detection_path="/data/detection_result.json"
tracking_path="/data/track_results/tracking_result.json"
vis_th=0.0
vis_img_bbox=1
init_idx=1000
python visualize.py --detection_path $detection_path --track_res_path $tracking_path --vis_th $vis_th --init_idx $init_idx \
--vis_img_bbox $vis_img_bbox
