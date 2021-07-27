#!/usr/bin/env python3

import os, sys
import numpy as np
import imageio
import time
import cv2
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import random
from nerf_helpers import *
from render_helpers import *
from utils import *
from load_data import load_data, load_pose_file, render_path_spiral

np.random.seed(0)
DEBUG = False

def config_parser():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='./configs/fern.yaml', help='config file (yaml) path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/cat', help='input data directory')
    parser.add_argument("--localdir", type=str, default='./logs', help='where to store local files')
    parser.add_argument("--testdir", type=str, default='./logs', help='where to store test files')
    parser.add_argument("--tblogdir", type=str, default='./logs', help="where to store tensorboard logs")
    parser.add_argument("--concat_times_views", action='store_true', help='concat times dimension with views dimension, otherwise concat time dimension with input pts dimension.')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=512, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=50, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 64, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')
    parser.add_argument("--w_depth", type=str, default=0, help='weight ratio for depth loss')
    parser.add_argument("--w_st", type=str, default=0, help='weight ratio for static loss')
    parser.add_argument("--w_zd", type=str, default=0, help='weight ratio for zero density loss')
    parser.add_argument("--N_iters", type=int, default=1000000, help='max number of training iterations')
    parser.add_argument("--train_frames", type=str, default='', help='frame numbers for training two-frame experiments')
    parser.add_argument("--test_frames", type=str, default='', help='frame numbers for testing two-frame experiments')
    parser.add_argument("--bandwidth_ratio_thres", type=float, default=0.05, help='default set to 0.05, the width of the band for applying static loss, should be adaptive to each specific scene')
    parser.add_argument("--depth_ratio_thres", type=float, default=1.03, help='default set to 1.03, threshold for labeling depth edges based on depth ratio on neighboring pixels, should be adaptive to each specific scene')

    # rendering options
    parser.add_argument('--use_times', action='store_true', help='use 4D input with time dimension instead of 3D')
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_times", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_dir", type=str, default='final_render', help='subdirectory that stores render results')
    parser.add_argument("--render_pose_list", type=str, default='render_pose.txt', help='filename of a specifica camera trajectory input')
    parser.add_argument("--render_start", type=int, default=0, help='specify start frame for rendering a video')
    parser.add_argument("--render_end", type=int, default=200, help='specify end frame for rendering a video')
    parser.add_argument("--render_type", type=str, default='space_time', help='filename of a specifica camera trajectory input')

    # dataset options
    parser.add_argument("--testskip", type=int, default=8, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--depth_stream", type=str, default='DF/1', help='set depth stream name')
    parser.add_argument("--color_stream", type=str, default='down', help='set color stream name')
    parser.add_argument("--down_factor", type=int, default=1, help='downsampling factor for spped up training, set 0 for 1080 * 1920 videos, 2 for 540 * 960 videos')
    parser.add_argument("--scale_factor", type=int, default=1, help='rescale colmap depth')
    parser.add_argument("--train_every_n", type=int, default=1, help='skip every n frames during training')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=200, help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=1000, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')

    # distributed learning options
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to be used")
    parser.add_argument("--load_multigpu", action='store_true', help="Force to use Multi GPU")
    parser.add_argument("--log_all_procs", action='store_true', help="Log from all processes insetad of just rank 0")

    return parser


def train(rank, args, log_dir):

    np.random.seed(rank)
    writer = SummaryWriter(os.path.join(log_dir, str(rank)))
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        if args.num_gpus > 1:
            raise ValueError("Can't run distributed training without GPUs")
        device = torch.device("cpu")

    if args.num_gpus > 1:
        dist.init_process_group("nccl", f"zeus://{args.expname}", world_size=args.num_gpus, rank=rank)

    # Load data
    images, depths, poses, times, hwf = load_data(args.datadir, args.down_factor, args.scale_factor)
    poses = poses[:, :3, :4]
    render_rgbs = images
    render_poses, render_intr = load_pose_file(os.path.join(args.datadir, args.render_pose_list))
    render_times = np.array(times)
    render_hwf = hwf
    min_depth = depths.min()
    max_depth = depths.max()
    print('Loaded cvd', images.shape, render_poses.shape, hwf, args.datadir)
    i_train = np.arange(images.shape[0])[::args.train_every_n]

    print('DEFINING BOUNDS')
    if args.no_ndc:
        print('no NDC ')
        near = np.min(depths)
        far = np.max(depths)
    else:
        print('NDC ')
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    # Cast intrinsics to right types
    H, W, fx, fy = hwf
    H, W = int(H), int(W)

    if args.render_only:
        if args.render_type ==  "space_time" and os.path.exists(os.path.join(args.datadir, args.render_pose_list)):
            print('Loading render poses from ', args.render_pose_list)
            render_poses, render_intr = load_pose_file(os.path.join(args.datadir, args.render_pose_list))
            render_times = np.array(times)
            render_hwf = hwf
        elif args.render_type ==  "fix_view":
            print('render from middle pose')
            middle_pose = int(len(poses)/2)
            render_poses = np.array(poses[np.repeat(middle_pose, len(poses))]) # Fix at middle frame
            render_hwf = hwf
            render_times = times
        elif args.render_type ==  "fix_time":
            print("render spiral motion at fix time frame")
            fix_time = 80
            render_poses = render_path_spiral(poses[fix_time-5*args.train_every_n:fix_time+5*args.train_every_n], hwf[-1], zrate=.5, rots=1, N=90) # Changing view
            render_times = np.array(times[np.repeat(fix_time, len(render_poses))]) # Fix time
        render_poses = np.stack(render_poses, 0)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    if not os.path.exists(os.path.join(basedir, expname)):
        os.makedirs(os.path.join(basedir, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.yaml')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, device)
    global_step = start

    bds_dict = {
        'near' : near * 0.8,
        'far' : far * 1.2,
        'min_depth' : min_depth,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():

            testsavedir = os.path.join(args.testdir, expname, 'render_{}_{:06d}'.format(args.render_type, global_step))
            if not os.path.exists(testsavedir):
                os.makedirs(testsavedir)
            print('save to dir:', testsavedir)
            print('hwf', hwf, 'render hwf', render_hwf)

            render_rgbs = torch.Tensor(render_rgbs).to(device)

            rgbs, disps = render_path(render_poses, render_times, render_hwf, render_rgbs, args.chunk, render_kwargs_test, gt_imgs=images, gt_depths=depths, savedir=testsavedir, render_factor=args.render_factor, render_start=args.render_start)
            if len(rgbs) > 0:
                with open(os.path.join(testsavedir, 'video_{}_{}.mp4'.format(args.render_start, args.render_end)), 'wb') as f:
                    imageio.mimwrite(f, to8b(rgbs), fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
                with open(os.path.join(testsavedir, 'disp_{}_{}.mp4'.format(args.render_start, args.render_end)), 'wb') as f:
                    imageio.mimwrite(f, disps, fps=30, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
                print('Done rendering', testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    print(args)
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        all_rays = np.stack([get_rays_np(hwf, p, t) for p, t in zip(poses[:,:3,:4], times)], 0) # [N, ro+rd+t, H, W, 3]
        # print('done, concats')
        all_rays = np.concatenate((all_rays, images[:,None]), 1) # [N, ro+rd+t+rgb+depth+flow, H, W, 3]
        all_rays = np.transpose(all_rays, [0,2,3,1,4]) # [N, H, W, ro+rd+t+rgb+depth+flow, 3]
        all_rays = np.stack([all_rays[i] for i in i_train], 0) # train images only
        all_rays = np.reshape(all_rays, [-1,4,3]) # [(N-1)*H*W, ro+rd+t+rgb+depth+flow, 3]
        all_rays = all_rays.astype(np.float32)

        all_depths = np.stack([depths[i] for i in i_train], 0)
        all_depths = np.reshape(all_depths, [-1,1]) # [N*H*W, 1]

        all_rays_o = all_rays[:,0,:]
        all_rays_d = all_rays[:,1,:]
        print('sample along rays')
        all_pts = sample_along_rays(rays_o=all_rays_o, rays_d=all_rays_d, depths=all_depths, near_depth=min_depth, far_depth=max_depth,
                N_samples=args.N_samples, inv_uniform=args.lindisp, perturb=args.perturb)
        testsavedir = os.path.join(args.testdir, expname, 'renderonly_{}_{:06d}'.format('path', start))
        if not os.path.exists(testsavedir):
            os.makedirs(testsavedir)
        point_cloud_path = os.path.join(testsavedir, 'point_cloud')
        if not os.path.exists(point_cloud_path):
            os.makedirs(point_cloud_path)
        all_pts_vis = np.ascontiguousarray(all_pts[::100, ::10, :].reshape((-1, 3)))
        random_colors = np.random.choice(range(255), size=3) / 255. * np.ones_like(all_pts_vis)

        print(all_pts_vis.shape)
        print('writing to ply', os.path.join(point_cloud_path, 'all_pts.ply'))
        write_ply(os.path.join(point_cloud_path, 'all_pts.ply'),
            x=all_pts_vis[:,0], y=-all_pts_vis[:,2], z=all_pts_vis[:,1], c=random_colors)

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    times = torch.Tensor(times).to(device)
    depths = torch.Tensor(depths).to(device)

    if use_batching:
        all_rays = torch.Tensor(all_rays).to(device)
        # all_depths = torch.Tensor(all_depths).to(device)
        # all_flows = torch.Tensor(all_flows).to(device)

    N_iters = args.N_iters
    print('Begin')
    print('TRAIN views are', i_train)

    if use_batching:
        print('shuffle rays')
        # np.random.shuffle(all_rays)
        rand_idx = torch.randperm(all_rays.shape[0])
        all_rays = all_rays[rand_idx]
        # all_depths = all_depths[rand_idx]
        # all_flows = all_flows[rand_idx]

    render_step = 0
    for i in range(start, N_iters):
        time0 = time.time()
        optimizer.zero_grad()
        loss = 0

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = all_rays[i_batch:(i_batch + args.N_rand)]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:3], batch[3]
            # target_d = all_depths[i_batch:(i_batch + args.N_rand)]
            # target_f = all_flows[i_batch:(i_batch + args.N_rand)]

            i_batch += args.N_rand
            if i_batch >= all_rays.shape[0]:
                tqdm.write("Shuffle data after an epoch!")
                rand_idx = torch.randperm(all_rays.shape[0])
                all_rays = all_rays[rand_idx]
                i_batch = 0

            #####  Core optimization loop  #####
            num_frames = float(images.shape[0])
            ret = render(num_frames, hwf, chunk=args.chunk, rays=batch_rays, depths=depths,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

        if N_rand is not None:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            t = times[img_i]

            if N_rand is not None:
                depth = depths[img_i]
                rays_o, rays_d, tf = get_rays(hwf, torch.Tensor(pose), torch.Tensor(t))  # (H, W, 3), (H, W, 3)
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                tf = tf[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1?)
                batch_rays = torch.stack([rays_o, rays_d, tf], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_d = depth[select_coords[:, 0], select_coords[:, 1]]

                if not args.depth_stream == 'depth_gt':
                    edge_map = compute_edge_map(depths[img_i], args.depth_ratio_thres)
                    edge_map = edge_map[select_coords[:, 0], select_coords[:, 1]]
                else:
                    edge_map = None

        #####  Core optimization loop  #####
        num_frames = float(images.shape[0])
        ret = render(num_frames, hwf, chunk=args.chunk, rays=batch_rays, depths=target_d,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        # radiance matching in static space query at random times
        if args.w_st > 0:
            static_mask = ret['static_mask'].unsqueeze(-1).repeat(1,1,3)
            rgb_static_loss = args.w_st * torch.sum(static_mask * torch.abs(ret['raw_rgb'] - ret['rgb_rand'])) / torch.sum(static_mask)
            static_mask = ret['static_mask']
            alpha_static_loss = args.w_st * torch.sum(static_mask * torch.abs(ret['raw_alpha'] - ret['alpha_rand'])) / torch.sum(static_mask)
            loss = loss + rgb_static_loss + alpha_static_loss

        # encourage zero volume density at the points before hitting the first surface along each ray.
        if args.w_zd > 0:
            near_mask = ret['near_mask']
            zero_density_loss = args.w_zd * torch.sum(near_mask * torch.abs(ret['raw_alpha'])) / torch.sum(near_mask)
            loss = loss + zero_density_loss

        # color loss

        img_loss = img2mse(ret['rgb_map'], target_s)
        loss = loss + img_loss

        # depth loss

        if not torch.isnan(target_d).any():
            if args.w_depth > 0:
                depth_loss = args.w_depth * compute_depth_loss(ret['depth_map'], target_d, edge_map, min_depth, args.no_ndc)
                loss = loss + depth_loss
        else:
            print(f"! [Numerical Error] {img_i} contains nan or inf.")

        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            img_loss0 = img2mse(ret['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

            if args.w_depth > 0:
                depth_loss_0 = args.w_depth *  compute_depth_loss(ret['depth0'], target_d, edge_map, min_depth, args.no_ndc)
                loss = loss + depth_loss_0

            # radiance matching in static space query at random times
            if args.w_st > 0:
                static_mask_0 = ret['static_mask_0'].unsqueeze(-1).repeat(1,1,3)
                rgb_static_loss_0 = args.w_st * torch.sum(static_mask_0 * torch.abs(ret['raw_rgb_0'] - ret['rgb_rand_0'])) / torch.sum(static_mask_0)
                static_mask_0 = ret['static_mask_0']
                alpha_static_loss_0 = args.w_st * torch.sum(static_mask_0 * torch.abs(ret['raw_alpha_0'] - ret['alpha_rand_0'])) / torch.sum(static_mask_0)
                loss = loss + rgb_static_loss_0 + alpha_static_loss_0

            # encourage zero volume density at the points before hitting the first surface along each ray.
            if args.w_zd > 0:
                near_mask_0 = ret['near_mask_0']
                zero_density_loss_0 = args.w_zd * torch.sum(near_mask_0 * torch.abs(ret['raw_alpha_0'])) / torch.sum(near_mask_0)
                loss = loss + zero_density_loss_0

        loss.backward()
        # NOTE: same as tf till here - 04/03/2020

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        print(f"Step: {global_step}, Loss: {loss}, lr: {new_lrate}, Time: {dt}")
        if args.w_depth > 0:
            print(f"img loss: {img_loss}, depth loss: {depth_loss}")
        if args.w_st > 0:
            print(f"rgb static loss: {rgb_static_loss}, alpha static loss: {alpha_static_loss}")
        if args.w_zd > 0:
            print(f"zero density loss: {zero_density_loss}")
        #####           end            #####

        # Rest is logging
        # saving checkpoints
        if i % args.i_weights == 0 and i > 0 and rank == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0:
                with open(path, "wb") as f:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f)
            else:
                with open(path, "wb") as f:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > 0 and rank == 0:

            with torch.no_grad():
                if os.path.exists(os.path.join(args.datadir, args.render_pose_list)):
                    print('Loading render poses from ', args.render_pose_list)

                    testsavedir = os.path.join(args.testdir, expname, 'renderonly_{}_{:06d}'.format('path', i))
                    if not os.path.exists(testsavedir):
                        os.makedirs(testsavedir)
                    print('save dir:', testsavedir)

                    rgbs, disps = render_path(render_poses, render_times, render_hwf, render_rgbs, args.chunk, render_kwargs_test, gt_imgs=None, gt_depths=None, savedir=testsavedir, render_factor=args.render_factor)
                    with open(os.path.join(testsavedir, 'video.mp4'), 'wb') as f:
                        imageio.mimwrite(f, to8b(rgbs), fps=15, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
                    with open(os.path.join(testsavedir, 'disp.mp4'), 'wb') as f:
                        imageio.mimwrite(f, disps, fps=15, quality=8, format='ffmpeg', output_params=["-f", "mp4"])
                    print('Saved test set', testsavedir)

        # Tensorboard visualization
        if i % args.i_print == 0 and (rank == 0 or args.log_all_procs):
            print('iter time {:.05f}'.format(dt))
            writer.add_scalar('loss', loss.detach().cpu(), global_step=global_step)
            writer.add_scalar('img_loss', img_loss.detach().cpu(), global_step=global_step)
            if args.w_depth > 0:
                writer.add_scalar('depth_loss', depth_loss.detach().cpu(), global_step=global_step)
            writer.add_scalar('loss', loss.detach().cpu(), global_step=global_step)
            if args.w_st > 0:
                writer.add_scalar('rgb_static_loss', rgb_static_loss.detach().cpu(), global_step=global_step)
                writer.add_scalar('alpha_static_loss', alpha_static_loss.detach().cpu(), global_step=global_step)
            if args.w_zd > 0:
                writer.add_scalar('zero_density_loss', zero_density_loss.detach().cpu(), global_step=global_step)
            if args.N_importance > 0:
                if args.w_st > 0:
                    writer.add_scalar('rgb_static_loss_0', rgb_static_loss_0.detach().cpu(), global_step=global_step)
                    writer.add_scalar('alpha_static_loss_0', alpha_static_loss_0.detach().cpu(), global_step=global_step)
                if args.w_zd > 0:
                    writer.add_scalar('zero_density_loss_0', zero_density_loss_0.detach().cpu(), global_step=global_step)

            if i % args.i_img == 0 and i > 0:
                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_train)
                if render_rgbs is not None:
                    target = images[img_i]
                    pose = render_poses[img_i,:3,:4]
                    t = render_times[img_i]
                    target_d = depths[img_i]
                    # edge_map = compute_edge_map(target_d, args.depth_ratio_thres)

                num_frames = float(images.shape[0])

                with torch.no_grad():
                    ret = render(num_frames, hwf, chunk=1024 * 32, c2w=pose, t=t, retraw=False, **render_kwargs_test)
                    img_loss = img2mse(ret['rgb_map'], target)
                    psnr_eval = mse2psnr(img_loss)
                    writer.add_scalar('psnr_train', psnr.detach().cpu(), global_step=global_step)
                    writer.add_scalar('psnr_eval', psnr_eval.detach().cpu(), global_step=global_step)

                    if args.no_ndc:
                        disp_vis = visualize_depth(1.0 / ret['depth_map'].detach().cpu().numpy()).astype(np.uint8)[..., ::-1].transpose(2, 0, 1)
                    else:
                        disp_vis = visualize_depth(1.0 - ret['depth_map'].detach().cpu().numpy()).astype(np.uint8)[..., ::-1].transpose(2, 0, 1)
                    gt_disp_vis = visualize_depth(1.0 / target_d.detach().cpu().numpy()).astype(np.uint8)[..., ::-1].transpose(2, 0, 1)
                    writer.add_image(args.render_type+'_rgb', to8b(ret['rgb_map'].detach().cpu().numpy()).transpose(2, 0, 1), global_step=global_step)
                    writer.add_image(args.render_type+'_depth', disp_vis, global_step=global_step)
                    writer.add_image("train_depth", gt_disp_vis, global_step=global_step)
                    writer.add_image('train_rgb', to8b(target.detach().cpu().numpy()).transpose(2, 0, 1), global_step=global_step)
                    # edge_map_vis = edge_map.unsqueeze(-1).repeat(1,1,3).detach().cpu().numpy().astype(np.uint8).transpose(2, 0, 1)
                    # writer.add_image('edge_map', edge_map_vis, global_step=global_step)

                    if args.N_importance > 0:
                        img_loss0 = img2mse(ret['rgb0'], target)
                        psnr0_eval = mse2psnr(img_loss0)
                        writer.add_scalar('psnr0_train', psnr0.detach().cpu(), global_step=global_step)
                        writer.add_scalar('psnr0_eval', psnr0_eval.detach().cpu(), global_step=global_step)

                        writer.add_image('rgb0', to8b(ret['rgb0'].detach().cpu().numpy()).transpose(2, 0, 1), global_step=global_step)
                        disp0_vis = visualize_depth(ret['disp0'].detach().cpu().numpy()).astype(np.uint8)[..., ::-1].transpose(2, 0, 1)
                        writer.add_image('disp0', disp0_vis, global_step=global_step)
                render_step += 1

        global_step += 1

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__=='__main__':
    parser = config_parser()
    args = config_parse_args(parser)

    summary_path = os.path.join(args.tblogdir, args.expname)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if args.num_gpus > 1:
        mp.spawn(train, nprocs=args.num_gpus, args=(args, summary_path))
    else:
        train(0, args, summary_path)
