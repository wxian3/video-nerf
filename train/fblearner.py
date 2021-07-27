#!/usr/bin/env python3

import argparse
from typing import Optional

from utils import train_fblearner


def run(name_prefix: Optional[str] = None, render_only: bool = False, train_stereo: bool = False,
        dataset_name: Optional[str] = None, num_gpus: int = 1, lrate: float = 0.0005, num_frames: int = 750,
        pose_list: Optional[str] = None, render_dir: Optional[str] = None):

    cvd_datasets = [
        "ayush_wave" , # Ayush.
        "VID_20190928_123018" ,  # Boy1.
        "VID_20190920_185325" ,  # Dog.
        "VID_20190919_202641" ,  # Cat2.
        "VID_20190920_184235" , # Person3
        "VID_20190922_190151" , # Boy2.
        "VID_20190928_151113" , # Person1.
        "VID_20190928_151303" , # Person2.
        "VID_20190929_215728" ,  # Cat.
        "VID_20190928_122603" , # Boy3
    ]

    adobe_datasets = [
        "AdobeStock_376645051" , # Girl typing.
        "AdobeStock_294965925" ,  # Parkour 1.
        "AdobeStock_268998415" , # Parkour 2.
        "AdobeStock_275529433" ,  # Outdoor jumping.
        "AdobeStock_206588697" ,  # Indoor jumping.
        "AdobeStock_205121267" , # Playing chess.
        "AdobeStock_356500789" , # Playing piano.
    ]

    sintel_datasets = [
        "bandage_1_final" ,
        "bandage_2_final" ,
        "sleeping_1_final" ,
        "sleeping_2_final" ,
        "alley_1_final" ,
        "bamboo_1_final" ,
    ]

    losses = {
        # "notime":                       ', "use_time": false, "N_rand" : 512', # Static NeRF
        # "rgb":                          ', "w_zd": 0, "w_sf": 0, "w_depth": 0, "N_rand" : 512',  # Temporal NeRF.
        # "rgb+depth":                      ', "w_zd": 0, "w_sf": 0, "w_depth": 1, "N_rand" : 512',  # + depth
        # "rgb+depth+zd":                 ', "w_zd": 1, "w_sf": 0, "w_depth": 1, "N_rand" : 512',  # + depth + zero density
        # "rgb+depth+static":             ', "w_zd": 0, "w_st": 1, "w_sf": 0, "w_depth": 1, "N_rand" : 512',  # + depth
        "rgb+depth+static+zd":            ', "w_zd": 0.1, "w_st": 0.1, "w_sf": 0, "w_depth": 0.2, "N_rand" : 512',  # NeRF + depth loss + static + zero density
        # "rgb+depth+flow+zd":            ', "w_zd": 1, "w_sf": 0.1, "w_depth": 1, "use_sf": true, "use_viewdirs": true, "N_rand" : 384',  # With both depth and scene flow losses + view direction.
        # "rgb+depth+static+flow+view+zd":', "w_zd": 1, "w_st": 10, "w_sf": 0.1, "w_depth": 1, "use_sf": true, "use_viewdirs": true, "N_rand" : 384',  # With both depth and scene flow losses + view direction.
    }

    if render_only:
        render_splits = []
        render_every_n = 450
        for i in range(0, num_frames, render_every_n):
            i_end = min(num_frames-1, i + render_every_n)
            render_splits.append(', "render_start": {}, "render_end": {}'.format(i, i_end))

    rendering_options = ', "render_only": true, "render_test": false, "render_type": "fix_time"'
    if pose_list and render_dir:
        rendering_options += ', "pose_list": "{}", "render_dir": "{}"'.format(pose_list, render_dir)

    # default options for model size
    training_options = ', "no_ndc": true, "lindisp": true, "netwidth": 512, "netwidth_fine": 1024, "N_samples" : 64, "N_importance": 128, "lrate_decay": 100, "lrate": 0.0005'
    if train_stereo:
        training_options += ', "train_stereo": true'

    training_options += f', "num_gpus": {num_gpus}'

    generic_config = "manifold://compphoto_data/tree/implicit/nerf/configs/generic_config.yaml"

    if dataset_name == 'cvd':
        data_basedir = "manifold://compphoto_data/tree/cvd_siggraph_2020"
        datasets = cvd_datasets
        training_options += ', "color_stream": "full", "depth_stream": "DF/1", "down_factor": 2'
    elif dataset_name == 'adobe':
        data_basedir = "manifold://compphoto_data/tree/AdobeStock_Videos"
        datasets = adobe_datasets
        training_options += ', "color_stream": "full", "depth_stream": "e0019", "scale_factor": 1000, "down_factor": 4, "train_every_n": 5, "pose_list": "pose_list.txt"'
        training_options += ', "bandwidth_ratio_thres": 0.05, "depth_ratio_thres": 1.03'
    elif dataset_name == 'sintel':
        data_basedir = "manifold://x3d_video3d/tree/static/Sintel_stereo"
        datasets = sintel_datasets
        training_options += ', "color_stream": "color_final_left", "depth_stream": "depth_left"'
    elif dataset_name == 'cvd2':
        data_basedir = "manifold://compphoto_data/tree/AdobeStock_Videos_CVD2"
        datasets = adobe_datasets
        training_options += ', "color_stream": "full", "depth_stream": "e0000_filtered", "scale_factor": 1000, "down_factor": 4, "train_every_n": 5, "pose_list": "pose_list.txt"'
        training_options += ', "bandwidth_ratio_thres": 0.05, "depth_ratio_thres": 1.03'
    else:
        raise RuntimeError(f"Unrecognized dataset name: {dataset_name}")

    for dataset in datasets:
        for loss_name, loss_options in losses.items():
            # PathManager.get_local_path() requires the trailing slash.
            datadir = f"{data_basedir}/{dataset}/"
            expname = (
                f"{name_prefix or 'x3d.nerf.training.workflow'}"
                f"_{dataset}_{loss_name}"
            )
            if render_only:
                for render_split in render_splits:
                    further_options = (
                        f"{render_split}"
                        f"{rendering_options}"
                        f"{training_options}"
                        f"{loss_options}"
                    )
                    params = (
                        f'{{"config": "{generic_config}", "expname": "{expname}"'
                        f', "datadir": "{datadir}"{further_options}}}'
                    )
                    train_fblearner(params, expname)
            else:
                further_options = (
                    f"{loss_options}"
                    f"{training_options}"
                )
                params = (
                    f'{{"config": "{generic_config}", "expname": "{expname}"'
                    f', "datadir": "{datadir}"{further_options}}}'
                )
                train_fblearner(params, expname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--train-stereo", action="store_true")
    parser.add_argument("--dataset-name")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--lrate", type=float, default=0.0005)
    parser.add_argument("--num-frames", type=int, default=450)
    parser.add_argument("--pose-list")
    parser.add_argument("--render-dir")
    args = parser.parse_args()

    run(**vars(args))
