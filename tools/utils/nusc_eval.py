from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
import argparse
import os

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def nusc_eval(res_path, eval_set="val", output_dir=None, data_path=None):

    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=data_path,
    )
    metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--input",
            required=True,
    )
    args = parser.parse_args()
    dataroot = 'data/nuscenes'
    root = 'data/Final_comp_results'
    result_path = os.path.join(root, args.input)

    id = args.input.split('.')[0]
    out_dir = os.path.join(root, 'eval')
    out_dir = os.path.join(out_dir, id)
    mkdir_or_exist(out_dir)
    print(f"Evaluating {id}...\n")
    nusc_eval(
            result_path,
            eval_set="val",
            output_dir=out_dir,
            data_path=dataroot,
    )
    