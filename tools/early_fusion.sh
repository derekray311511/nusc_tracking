
config="configs/early_fusion.yaml"
workspace="/home/Student/Tracking"
out_dir="/data/early_fusion_track_results/avgScore_segDetThr0.0_segDistThr0"
evaluate=1

python tools/early_fusion.py $config --workspace $workspace --out-dir $out_dir --evaluate $evaluate --out_time --save_log