#!/usr/bin/env python3
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_times=1, output_ch=4, skips=[4], use_viewdirs=False, use_times=False, concat_times_views=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_times = input_ch_times
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.use_times = use_times
        self.concat_times_views = concat_times_views

        if self.use_times and not self.concat_times_views:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch + input_ch_times, W)]
                + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch + input_ch_times, W)
                    for i in range(D - 1)
                ]
            )

        else:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch, W)]
                + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                    for i in range(D - 1)
                ]
            )

        views_linears_input_ch = 0
        if self.use_viewdirs:
            views_linears_input_ch += input_ch_views
        if self.use_times and self.concat_times_views:
            views_linears_input_ch += input_ch_times

        if views_linears_input_ch > 0:
            self.views_linears = nn.ModuleList([nn.Linear(views_linears_input_ch + W, W // 2)])

        if self.use_viewdirs or (self.use_times and self.concat_times_views):
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
            self.feature_linear = nn.Linear(W, W)
        else:
            self.output_linear = nn.Linear(W, output_ch)


    def forward(self, x):
        input_pts, input_views, input_times = torch.split(x, [self.input_ch, self.input_ch_views, self.input_ch_times], dim=-1)

        if self.use_times and not self.concat_times_views:
            h = torch.cat([input_pts, input_times], -1)
        else:
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                if self.use_times and not self.concat_times_views:
                    h = torch.cat([input_pts, input_times, h], -1)
                else:
                    h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs or (self.use_times and self.concat_times_views):
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            feats = [feature]
            if self.use_viewdirs:
                feats.append(input_views)
            if self.use_times and self.concat_times_views:
                feats.append(input_times)
            h = torch.cat(feats, -1)


            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        assert self.use_times, "Not implemented if use_times=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))



# Ray helpers
def get_rays(hwf, c2w, time):
    H, W, fx, fy = hwf
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/fx, -(j-H*.5)/fy, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    times = time.expand_as(rays_o)
    return rays_o, rays_d, times


def get_rays_np(hwf, c2w, time):
    H, W, fx, fy = hwf
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/fx, -(j-H*.5)/fy, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    times = np.full(rays_o.shape, time)
    return rays_o, rays_d, times


def ndc_rays(hwf, near, rays_o, rays_d):
    H, W, fx, fy = hwf
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*fx)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*fy)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*fx)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*fy)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def img2mse(x, y):
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / (SSIM_d)

    return torch.clamp((SSIM), 0, 1)


def compute_ssim(x, y):

    ssim = torch.mean(SSIM(x, y))
    return ssim

def compute_psnr(x, y, mask):
    mse = torch.sum(mask * (x - y) ** 2) / torch.sum(mask)
    return mse2psnr(mse)
    
def compute_scale_invariant_depth_loss(pred_depth, gt_depth):
    pred_disp = 1.0 / pred_depth
    gt_disp = 1.0 / gt_depth
    t_pred = torch.median(pred_disp)
    s_pred = torch.mean(torch.abs(pred_disp - t_pred))

    t_gt = torch.median(gt_disp)
    s_gt = torch.mean(torch.abs(gt_disp - t_gt))

    pred_disp_n = (pred_disp - t_pred) / s_pred
    gt_disp_n = (gt_disp - t_gt) / s_gt

    return torch.mean(torch.abs(pred_disp_n - gt_disp_n))

def NDC2Euclidean(z_ndc, gt_depth, min_depth):
    z_e = 1. * min_depth / (1. - z_ndc)
    return z_e

def compute_ratio(x, y):
    d_min = torch.min(x, y)
    d_max = torch.max(x, y)

    return d_max / (d_min + 1e-10)


def depth_ratio(x):
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    x = x.view(1, h_x, w_x)
    # gradient step=1
    right = F.pad(x, [0, 1, 0, 0])[:, :, 1:]
    bottom = F.pad(x, [0, 0, 0, 1])[:, 1:, :]

    ratio_h = compute_ratio(right, x)  # x: left
    ratio_v = compute_ratio(bottom, x)  # x: top

    # h_ratio will always have zeros in the last column, right-left
    # v_ratio will always have zeros in the last row,    bottom-top
    ratio_h[:, :, -1] = 0
    ratio_v[:, -1, :] = 0

    return (ratio_h[0], ratio_v[0])


def compute_edge_map(depth, depth_ratio_thresh):
    # gradient step=1
    ratio_pred_h, ratio_pred_v  = depth_ratio(depth / torch.median(depth))
    edge_map_h = ratio_pred_h < depth_ratio_thresh
    edge_map_v = ratio_pred_v < depth_ratio_thresh
    edge_map = torch.logical_and(edge_map_h, edge_map_v)

    return edge_map

def compute_depth_loss(pred_depth, gt_depth, edge_map, min_depth, no_ndc):
    if not no_ndc:
        pred_depth = NDC2Euclidean(pred_depth, gt_depth, min_depth)

    if torch.isnan(gt_depth).any() or torch.isnan(pred_depth).any():
        print("! [Numerical Error] depth map contains nan or inf.")
        return
    eps = 1e-8
    pred_disp = 1./torch.max(eps * torch.ones_like(pred_depth), pred_depth)
    gt_disp = 1./torch.max(eps * torch.ones_like(gt_depth), gt_depth)
    # if edge_map is not None:
    #     return torch.sum(torch.abs(edge_map * pred_disp - edge_map * gt_disp)) / torch.sum(edge_map)
    # else:
    #     return torch.mean(torch.abs(pred_disp - gt_disp))
    return torch.mean(torch.abs(pred_disp - gt_disp))


def sample_along_rays(rays_o, rays_d, depths, near_depth, far_depth, N_samples, inv_uniform, perturb):
    '''
    function for sampling along camera rays
    :param rays_o: origin of the ray
    :param rays_d: ray direction vectors
    :param inv_uniform, uniformly sampling in inverse depth
    :param perturb: randomly jittering the sampling positions
    '''

    near, far = near_depth * np.ones_like(rays_d[...,:1]), far_depth * np.ones_like(rays_d[...,:1])

    if inv_uniform:
        start = 1. / near
        step = (1. / far - start) / (N_samples -1)
        z_vals = 1. / np.stack([start+i*step for i in range(N_samples)], axis=1)
    else:
        start = near
        step = (far - near) / (N_samples-1)
        z_vals = np.stack([start+i*step for i in range(N_samples)], axis=1)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = np.concatenate((mids, z_vals[...,-1:]), axis=-1)
        lower = np.concatenate((z_vals[...,:1], mids), axis=-1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    print('depths', depths.shape)
    step = (far - near) / (N_samples-1)
    below = np.minimum(depths[:, np.newaxis, :] - 3 * step, z_vals)
    above = np.maximum(depths[:, np.newaxis, :] + 3 * step, z_vals)
    z_vals = np.concatenate((below, above), axis=-1)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals

    return pts
