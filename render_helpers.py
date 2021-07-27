#!/usr/bin/env python3
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
import os, sys
import numpy as np
import imageio
import time
import cv2
import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from nerf_helpers import *
import torch.distributed as dist

np.random.seed(0)
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, times, fn, embed_fn, embeddirs_fn, embedtimes_fn, use_times, use_viewdirs, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if use_viewdirs:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    if use_times:
        input_times = times[:,None].expand(inputs.shape[:2] + (1,))
        input_times_flat = torch.reshape(input_times, [-1, input_times.shape[-1]])
        embedded_times = embedtimes_fn(input_times_flat)
        embedded = torch.cat([embedded, embedded_times], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(num_frames, rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(num_frames, rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(num_frames, hwf, chunk=1024 * 32, rays=None, depths=None, c2w=None, t=None, ndc=True,
                  near=0., far=1., min_depth = 1.,
                  use_viewdirs=False, use_times=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      num_frames: total number of image frames.
      hwf: array of shape [1, 4]. Image height and width and focal lengths.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      t: the time dimension.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in ndc coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d, times = get_rays(hwf, c2w, t)
    else:
        # use provided ray batch
        rays_o, rays_d, times = rays

    # provide ray directions as input
    viewdirs = rays_d
    if c2w_staticcam is not None:
        # special case to visualize effect of viewdirs
        rays_o, rays_d, times = get_rays(hwf, c2w_staticcam, t)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(hwf, min_depth, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    times = torch.reshape(times[...,0], [-1,1]).float()
    if depths is not None:
        depths = torch.reshape(depths, [-1,1]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    rays = torch.cat([rays, viewdirs, times], -1)
    if depths is not None:
        rays = torch.cat([rays, depths], -1)

    # Render and reshape
    all_ret = batchify_rays(num_frames, rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # k_extract = ['rgb_map', 'disp_map', 'acc_map']
    # ret_list = [all_ret[k] for k in k_extract]
    # ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    # return ret_list + [ret_dict]
    return all_ret


def render_path(render_poses, render_times, render_hwf, render_rgbs, chunk, render_kwargs, gt_imgs=None, gt_depths=None, savedir=None, render_factor=0, render_start=0):

    H, W, fx, fy = render_hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        fx = fx / render_factor
        fy = fy / render_factor
        render_hwf = [H, W, fx, fy]

    rgbs = []
    disps = []
    disps_vis = []


    num_frames = render_poses.shape[0]
    t = time.time()
    if not os.path.exists(os.path.join(savedir, 'images')):
        os.makedirs(os.path.join(savedir, 'images'))
    if not os.path.exists(os.path.join(savedir, 'depths')):
        os.makedirs(os.path.join(savedir, 'depths'))

    for i, (c2w, render_time) in enumerate(zip(tqdm(render_poses), render_times)):
        imgs_savepath = os.path.join(savedir, 'images/{:03d}.png'.format(i+render_start))
        depths_savepath = os.path.join(savedir, 'depths/{:03d}.png'.format(i+render_start))

        print(i, time.time() - t)
        t = time.time()

        ret = render(num_frames, render_hwf, chunk=1024 * 32, c2w=c2w[:3, :4], t=render_time, retraw=False, **render_kwargs)
        rgb = ret['rgb_map'].cpu().numpy()
        disp = ret['disp_map'].cpu().numpy()
        depth = ret['depth_map'].cpu().numpy()

        depth_e = 1.0 - depth
        rgbs.append(rgb)
        disps.append(depth_e)
        disp_vis = visualize_depth(depth_e).astype(np.uint8)[..., ::-1]
        disps_vis.append(disp_vis)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        with open(imgs_savepath, 'wb') as f:
            plt.imsave(f, to8b(rgb), format='png')
            plt.close()
        with open(depths_savepath, 'wb') as f:
            plt.imsave(f, disp_vis, format='png')
            plt.close()

    if len(rgbs) > 0:
        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

    return rgbs, disps_vis


def create_nerf(args, device):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, input_dims=3)

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, input_dims=3)

    input_ch_times = 0
    embedtimes_fn = None
    if args.use_times:
        embedtimes_fn, input_ch_times = get_embedder(args.multires_times, args.i_embed, input_dims=1)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                 input_ch_times=input_ch_times, use_times=args.use_times,
                 concat_times_views=args.concat_times_views).to(device)
    if args.num_gpus > 1:
        model = DDP(model, device_ids=[device], broadcast_buffers=False)
    elif args.load_multigpu:
        model = DP(model, device_ids=[device])
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                          input_ch_times=input_ch_times, use_times=args.use_times,
                          concat_times_views=args.concat_times_views).to(device)
        if args.num_gpus > 1:
            model_fine = DDP(model_fine, device_ids=[device], broadcast_buffers=False)
        elif args.load_multigpu:
            model = DP(model, device_ids=[device])
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, times, network_fn : run_network(inputs, viewdirs, times, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtimes_fn=embedtimes_fn,
                                                                use_times=args.use_times,
                                                                use_viewdirs=args.use_viewdirs,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    localdir = args.localdir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1

        # Load model
        if dist.is_initialized():
            model.load_state_dict(ckpt['network_fn_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        else:
            model.load_state_dict(strip_module_prefix(ckpt['network_fn_state_dict']))
            optimizer.load_state_dict(strip_module_prefix(ckpt['optimizer_state_dict']))
            if model_fine is not None:
                model_fine.load_state_dict(strip_module_prefix(ckpt['network_fine_state_dict']))

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'bandwidth_ratio' : args.bandwidth_ratio_thres,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'use_times': args.use_times,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # ndc only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2alpha(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    raw2a = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2a(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    return alpha


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """ A helper function for `render_rays`.
    """
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    alpha = raw2alpha(raw, z_vals, rays_d, raw_noise_std, pytest)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    transmit = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmit
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(num_frames,
                ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                bandwidth_ratio=0.05,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,8:11] if ray_batch.shape[-1] > 8 else None
    times = ray_batch[:, 11:12] if ray_batch.shape[-1] > 11 else None
    depths = ray_batch[:, 12:13] if ray_batch.shape[-1] > 12 else None

    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    bandwidth = bandwidth_ratio * (far - near)

    rand_idx = random.randint(0, int(num_frames))
    rand_times = torch.ones_like(times) * rand_idx *  2. / num_frames - 1.
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    pts = pts[:, :, :3]
    raw = network_query_fn(pts, viewdirs, times, network_fn)
    raw_rgb = torch.sigmoid(raw[...,:3])
    raw_alpha = raw2alpha(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if depths is not None:
        # static constraints on areas not covered by CVD
        depth_near_bound = (depths - bandwidth).expand(z_vals.shape)
        depth_far_bound = (depths + bandwidth).expand(z_vals.shape)
        near_mask = z_vals < depth_near_bound
        far_mask = z_vals > depth_far_bound
        static_mask = torch.logical_or(far_mask, near_mask)

        raw_rand = network_query_fn(pts, viewdirs, rand_times, network_fn)
        rgb_rand = torch.sigmoid(raw_rand[...,:3])
        alpha_rand = raw2alpha(raw_rand, z_vals, rays_d, raw_noise_std, pytest=pytest)


    if N_importance > 0:
        # save outputs from coarse network
        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, raw_rgb_0, raw_alpha_0 = rgb_map, disp_map, acc_map, depth_map, raw_rgb, raw_alpha

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        pts = pts[:, :, :3]
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, times, run_fn)
        raw_rgb = torch.sigmoid(raw[...,:3])
        raw_alpha = raw2alpha(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        if depths is not None:
            # static constraints on areas not covered by CVD
            static_mask_0, near_mask_0, rgb_rand_0, alpha_rand_0 = static_mask, near_mask, rgb_rand, alpha_rand
            depth_near_bound = (depths - bandwidth).expand(z_vals.shape)
            depth_far_bound = (depths + bandwidth).expand(z_vals.shape)
            near_mask = z_vals < depth_near_bound
            far_mask = z_vals > depth_far_bound
            static_mask = torch.logical_or(far_mask, near_mask)

            raw_rand = network_query_fn(pts, viewdirs, rand_times, network_fn)
            rgb_rand = torch.sigmoid(raw_rand[...,:3])
            alpha_rand = raw2alpha(raw_rand, z_vals, rays_d, raw_noise_std, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map}

    if retraw:
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map, 'weights' : weights,
            'raw_rgb': raw_rgb, 'raw_alpha': raw_alpha, 'rgb_rand': rgb_rand, 'alpha_rand': alpha_rand, 'static_mask': static_mask, 'near_mask': near_mask}

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['depth0'] = depth_map_0
        ret['raw_rgb_0'] = raw_rgb_0
        ret['raw_alpha_0'] = raw_alpha_0
        if depths is not None:
            ret['rgb_rand_0'] = rgb_rand_0
            ret['alpha_rand_0'] = alpha_rand_0
            ret['static_mask_0'] = static_mask_0
            ret['near_mask_0'] = near_mask_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret
