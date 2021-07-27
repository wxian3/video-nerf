import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
import yaml
import cv2


def visualize_depth(depth, depth_min=None, depth_max=None):
    """Visualize the depth map with colormap.

    Rescales the values so that depth_min and depth_max map to 0 and 1,
    respectively.
    """
    if depth_min is None:
        depth_min = np.amin(depth)

    if depth_max is None:
        depth_max = np.amax(depth)

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = depth_scaled ** 0.5
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)

    return ((cv2.applyColorMap(
        depth_scaled_uint8, cv2.COLORMAP_MAGMA) / 255) ** 2.2) * 255


def strip_module_prefix(
    state_dict: OrderedDictType[str, torch.Tensor]
) -> OrderedDictType[str, torch.Tensor]:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            # Strip the prefix added by Data Parallel.
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def config_parse_args(parser, arg_strings=None):
    args = parser.parse_args(args=arg_strings)
    if args.config:
        print('Parsed config yaml file')
        with open(args.config, 'rb') as f:
            config_file = f.read()
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
            arg_dict = args.__dict__
            for key, value in configs.items():
                arg_dict[key] = value
    return args

def write_ply(
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: np.ndarray
) -> None:
    ''' write a ply file containing the given point cloud
    '''
    with open(filename, 'w') as f:
        n_pts = x.size

        # convert color values to uint8
        c = np.clip(np.round(c * 255), 0, 255).astype(np.uint8)

        # write the ply header
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {:d}\n'.format(n_pts))
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        # write vertices
        for x, y, z, (r, g, b) in zip(x, y, z, c):
            f.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(x, y, z, r, g, b))

        f.close()
