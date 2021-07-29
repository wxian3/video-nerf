import numpy as np
import math
import os
import sys
import struct
import cv2
import imageio
import matplotlib.pyplot as plt

def load_raw_float32_image(file_name):
    with open(file_name, "rb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5
        I_BYTES = 4
        Q_BYTES = 8

        h = struct.unpack("i", f.read(I_BYTES))[0]
        w = struct.unpack("i", f.read(I_BYTES))[0]

        cv_type = struct.unpack("i", f.read(I_BYTES))[0]
        pixel_size = struct.unpack("Q", f.read(Q_BYTES))[0]
        d = ((cv_type - CV_32F) >> CV_CN_SHIFT) + 1
        assert d >= 1
        d_from_pixel_size = pixel_size // 4
        if d != d_from_pixel_size:
            raise Exception(
                "Incompatible pixel_size(%d) and cv_type(%d)" % (pixel_size, cv_type)
            )
        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")

        data = np.frombuffer(f.read(), dtype=np.float32)
        result = data.reshape(h, w) if d == 1 else data.reshape(h, w, d)
        return result


def read_disp(disp_file, resize=None):
    disp = load_raw_float32_image(disp_file)
    if resize is not None:
        disp = cv2.resize(disp, resize, interpolation=cv2.INTER_AREA)
    return disp

def load_pose_file(fn):
    with open(fn, 'r') as f:
        extrinsics = []
        intrinsics = []
        for line in f.readlines():
            line = np.array(list(map(float, line.split())))
            pose = np.zeros((3, 4), dtype=np.float32)
            if len(line) == 16:
                halfW = int(line[0] / 2.0)
                halfH = int(line[1] / 2.0)
                vF = math.radians(line[15])
                pose[:, 0] = np.array(line[6:9]) # right
                pose[:, 1] = np.array(line[9:12]) # up
                pose[:, 2] = np.array(-line[12:15]) # backward
                pose[:, 3] = np.array(line[3:6]) # position
                extrinsics.append(pose)
                intrinsics.append(vF)
        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        f.close()

    return extrinsics, intrinsics


def load_data(local_base_dir, down_factor, scale_factor):

    # load images
    print('Loading data from ', local_base_dir)
    img_dir = os.path.join(local_base_dir, 'color_full')

    if os.path.exists(img_dir):
        img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith('.png')]
        imgs_rgb = [imageio.imread(f)/255. for f in img_files]
        imgs_rgb = np.stack(imgs_rgb, 0)

    img_height = int(imgs_rgb[0].shape[0] / down_factor)
    img_width = int(imgs_rgb[0].shape[1] / down_factor)

    imgs_rgb = [cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA) for img in imgs_rgb]
    imgs_rgb = np.stack(imgs_rgb, 0)

    # load camera poses
    N = imgs_rgb.shape[0]
    pose_file = os.path.join(local_base_dir, 'input_pose.txt')
    extrinsics, intrinsics = load_pose_file(pose_file)
    poses = np.zeros((N, 4, 4), dtype=np.float32)
    poses[:, :3, :4] = extrinsics[:N, :, :]
    # store (fx, fy, cx, cy) in the last row
    halfH, halfW = img_height/2., img_width/2.
    fy = halfH / math.tan(float(intrinsics[-1])/ 2.0)
    poses[:, -1, :4] = [fy, fy, halfH, halfW]
    hwf = [img_height, img_width, fy, fy]

    # read disparity maps
    disp_dir = os.path.join(local_base_dir, 'depth')
    if os.path.exists(disp_dir):
        disp_files = [os.path.join(disp_dir, f) for f in sorted(os.listdir(disp_dir)) if f.endswith('.raw')]
        disps = [read_disp(f, (img_width, img_height)) for f in disp_files]
        disps = np.stack(disps, 0)
    else:
        print('depth folder not found.')
        sys.exit()
        
    depths = 1.0 / disps

    # define input times
    times = np.linspace(-1, 1, poses.shape[0], dtype=np.float32).reshape(-1, 1)

    # rescale camera translation for colmap depth
    poses[:, :3, 3] = poses[:, :3, 3] * scale_factor

    print('Data Shapes:')
    print(poses.shape, imgs_rgb.shape, disps.shape)

    return imgs_rgb, depths, poses, times, hwf


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(poses, focal, zrate, rots, N):
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))
    # Get radii for spiral path
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 20, 0)
    c2w = poses_avg(poses)
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
