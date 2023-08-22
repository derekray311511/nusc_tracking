from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
import argparse
import os

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def nusc_eval(res_path, eval_set="val", out_dir=None, dataroot=None, custom_range=None, dist_th_tp=None, **kwargs):

    cfg = track_configs("tracking_nips_2019")
    if dist_th_tp is not None:
        print(f"Match dist is changed from {cfg.dist_th_tp} m ", end='')
        cfg.dist_th_tp = dist_th_tp
        print(f"to {cfg.dist_th_tp} m")

    if custom_range is not None:
        for k, v in cfg.class_range.items():
            cfg.class_range[k] = custom_range[k]

    print(f"Evaluate ranges for classes:")
    mkdir_or_exist(out_dir)
    info_path = os.path.join(out_dir, 'info.txt')
    with open(info_path, 'w') as f:
        f.write(f"dataroot: {dataroot}\n")
        f.write(f"dist_th_tp: {cfg.dist_th_tp}\n")
        for k, v in cfg.class_range.items():
            print(f"{k}: {v}")
            f.write(f"{k}: {v}\n")

    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=out_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=dataroot,
    )
    metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="/data/small_data2")
    parser.add_argument("--version", required=True)
    parser.add_argument("--range", type=int, default=None)
    parser.add_argument("--tp_dist", type=float, default=None)
    args = parser.parse_args()

    dataroot = args.dataroot
    result_path = os.path.join('/data/early_fusion_track_results', args.version)

    if args.range is None:
        range = "default"
        custom_range = None
    else:
        range = args.range
        custom_range = {
            "car": range,
            "truck": range,
            "bus": range,
            "trailer": range,
            "pedestrian": range,
            "motorcycle": range,
            "bicycle": range
        }
    out_dir = os.path.join(result_path, f"eval_{str(range)}_{int(args.tp_dist)}_998")
    mkdir_or_exist(out_dir)
    print(f"\nResult path: {result_path}")
    print(f"\nEvaluating {args.version} using dataset from '{dataroot}'...\n")
    nusc_eval(
            os.path.join(result_path, 'tracking_result.json'),
            eval_set="val",
            out_dir=out_dir,
            dataroot=dataroot,
            custom_range=custom_range,
            dist_th_tp=args.tp_dist,
    )
    print(f"Done. With dataset: \"{dataroot}\"")
