result_file='track_result_bboxth-0.05'
detection_path="/data/${result_file}/detection_result.json"
tracking_path="/data/${result_file}/tracking_result.json"
python visualize.py --detection_path ${detection_path} --track_res_path ${tracking_path}
