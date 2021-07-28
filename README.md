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

including camera poses and depth maps from CVD. Place the dataset under `./data/{DATASET}` and save the config file as `./configs/{DATASET}.yaml`.
Download the data for a video presented in our paper (e.g. replace `{DATASET}` with cat):
```
bash datasets/download_datasets.sh {DATASET}
```
This includes the video frames, camera poses and depth maps, and will be saved to `./datasets/{DATASET}`

# Demo

To train video NeRF on different datasets:
```
python run_nerf.py --config configs/{DATASET}.yaml
```
For example, raplace `{DATASET}` with cat_1, create a config file based on the example `./configs/example.yaml`, give a unique expname and specify base directory where logs are stored.

Training takes 24-48 hours using 2 NVIDIA V100 GPUs. Run tensorboard to visualize the training process:
```
tensorboard --logdir ./logs --port 8090
```

To test and render video NeRF on different datasets:
```
python run_nerf.py --config configs/{DATASET}.yaml --render_only
```
The results will rendered in spiral motion using a camera trajectory from `./data/{DATASET}/render_pose.txt`, then it will be saved in `logs/{expname}` folder as default. 

# Pre-trained Models

Download the pre-trained models:
```
bash datasets/download_pretrained_models.sh {DATASET}
```
Place the downloaded models in ./logs/{expname}/{CHECKPOINT}.tar in order to load it.

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
