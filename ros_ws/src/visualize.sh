detection_path="/data/detection_result.json"
tracking_path="/data/track_results/tracking_result.json"
vis_th=0.025
init_idx=0
python visualize.py --detection_path $detection_path --track_res_path $tracking_path --vis_th $vis_th --init_idx $init_idx
