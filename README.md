# Video-NeRF
PyTorch implementation of paper "Space-time Neural Irradiance Fields for Free-Viewpoint Video"

[[Project Website]](https://video-nerf.github.io/)
[[Paper]](https://arxiv.org/abs/2011.12950)
[[Video]](https://www.youtube.com/watch?v=2tN8ghNu2sI&t=1s)

<img src='teaser.gif' height="260px"/>

# Dependencies
Install PyTorch 1.8 and other dependencies, type command line:
```
pip install -r requirements.txt
```

# Dataset

You can download eight videos presented in the paper from [here](https://drive.google.com/drive/folders/1jghs7A0OLiYyyTrW5fEt6h4IQigFd2fP?usp=sharing), including camera poses and depth maps from CVD. Place the dataset under `./data/{DATASET}` and save the config file as `./configs/{DATASET}.yaml`.

# Demo

To train video NeRF on different datasets:
```
python train/run_nerf.py --config configs/{DATASET}.yaml
```
For example, raplace `{DATASET}` with cat

Training takes 24-48 hours using 2 NVIDIA V100 GPUs.

To test and render video NeRF on different datasets:
```
python train/run_nerf.py --config configs/{DATASET}.yaml --render_only
```
The results will be saved in `logs/{DATASET}` folder.

# Pre-trained Models

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1Gv5M_1D0gPmfaC74nzWooJfxabVu6sxW?usp=sharing). Place the downloaded models in ./logs/{DATASET}/{CHECKPOINT}.tar in order to load it.

# Custom Video

If you want to run on your own video, follow these steps:
1. Extract frames from your video and save them in `my_video/color_full`
``
mkdir ./data/my_video && cd ./datasets/my_video 
mkdir color_full && ffmpeg -i video.mp4 rgb/%06d.png
``
2. Run *COLMAP* to compute poses and save them to `my_video/input_pose.txt` in ... format (TODO)
3. Compute depth maps from [CVD](https://github.com/facebookresearch/consistent_depth) or other monocular video depth estimation method. Then save it to `my_video/depth`
4. Prepare a config file and save it to `configs/my_video.yaml`

# License 

This work is licensed under MIT License.
The code is based on implementation of [NeRF](https://github.com/yenchenlin/nerf-pytorch).

# Citation

If you find the code/models helpful, please consider to cite:
```
@inproceedings{xian2021space,
  title={Space-time neural irradiance fields for free-viewpoint video},
  author={Xian, Wenqi and Huang, Jia-Bin and Kopf, Johannes and Kim, Changil},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9421--9431},
  year={2021}
}
```
