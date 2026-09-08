
import math
import numpy as np

import torch
import torch.nn as nn
import platform

from torch import nn as nn

from LM_Tools.configuration.configuration import Configuration

# to be used as decorator for platform-dependent compiler exceptions
compile_disable_if_windows = (
    torch.compiler.disable
    if platform.system() == "Windows"
    else lambda f: f
)


# This is from an implementation of the T2T-ViT
# https://github.com/yitu-opensource/T2T-ViT/blob/main/models/transformer_block.py#L78
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def get_raw_patch_neighbors_mask(hor_patches, ver_patches, neighborhood):
    seq_length = hor_patches * ver_patches
    mask = torch.zeros((seq_length, seq_length))
    mask[:, :] = -torch.inf
    for p_y in range(ver_patches):
        for p_x in range(hor_patches):
            ref_patch = p_y * hor_patches + p_x
            # for patch at (p_y, p_x), find neighbors ...
            for n_y in range(max(0, p_y - neighborhood), min(ver_patches, p_y + neighborhood + 1)):
                for n_x in range(max(0, p_x - neighborhood), min(hor_patches, p_x + neighborhood + 1)):
                    other_patch = n_y * hor_patches + n_x
                    mask[ref_patch, other_patch] = 0.0

    return mask


def get_1d_sincos_pos_embed(embed_dim, positions):
    """
    Compute 1D sinusoidal embeddings
    positions: [N]
    """
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1. / (10000 ** (omega / (embed_dim / 2)))

    pos = positions.unsqueeze(1)  # [N, 1]
    out = pos * omega.unsqueeze(0)  # [N, D/2]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [N, D]
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Helper to compute sin/cos positional embeddings from a grid.
    grid: [2, N] where grid[0] = y coordinates, grid[1] = x coordinates
    """
    assert embed_dim % 2 == 0
    half_dim = embed_dim // 2

    # Half for height (y), half for width (x)
    emb_y = get_1d_sincos_pos_embed(half_dim, grid[0])  # [N, half_dim]
    emb_x = get_1d_sincos_pos_embed(half_dim, grid[1])  # [N, half_dim]
    return torch.cat([emb_y, emb_x], dim=1)  # [N, D]


def get_2d_sincos_pos_embed(embed_dim, offset_x, offset_y, grid_h, grid_w):
    """
    Generate 2D sinusoidal positional embeddings (ViT-style)
    Args:
        embed_dim: total embedding dimension (must be even)
        grid_h, grid_w: number of patches in height and width
    Returns:
        pos_embed: [grid_h * grid_w, embed_dim]
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even for sin/cos"

    # Create grid coordinates
    grid_y = torch.arange(grid_h, dtype=torch.float32) + offset_y
    grid_x = torch.arange(grid_w, dtype=torch.float32) + offset_x
    grid = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [2, H, W]

    # Stack and flatten -> [H*W, 2]
    grid = torch.stack(grid, dim=0).reshape(2, -1)

    # Compute 1D encodings and concatenate
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_relative_positional_embed(embed_dim, crop_Y, crop_X, crop_H, crop_W, img_H, img_W,
                                            temperature=10000.0, for_NAT=False):
    """
    Create a 2D sinusoidal positional encoding that generalizes to any resolution.

    Args:
        embed_dim: embedding dimension (must be divisible by 4)
        crop_X, crop_Y:  coordinate of the top-left patch of current image crops
        crop_H, crop_W:  spatial resolution of current image crops (#patches vertically, horizontally)
        img_H, img_W: spatial resolution of current image (#patches vertically, horizontally)
        temperature: frequency scaling (default 10000, same as original Transformer)
    Returns:
        pos_emb: (1, dim, H, W)
    """
    if embed_dim % 4 != 0:
        raise ValueError("Embedding dim must be divisible by 4 for 2D sin/cos encoding.")

    # Coordinate grid
    grid_y = torch.arange(crop_H, dtype=torch.float32, device=crop_Y.device) + crop_Y
    grid_x = torch.arange(crop_W, dtype=torch.float32, device=crop_Y.device) + crop_X
    y, x = torch.meshgrid(grid_y, grid_x, indexing="ij")

    # Scale coordinates from [0, img_Patches -1] to [0, 1]
    y = y / (img_H - 1)
    x = x / (img_W - 1)

    # Scale coordinates from [0, 1] to [0, 2π]
    y = y * 2 * math.pi
    x = x * 2 * math.pi

    dim_quarter = embed_dim // 4

    # Frequency bands
    omega = torch.arange(dim_quarter, device=crop_Y.device) / dim_quarter
    omega = 1.0 / (temperature ** omega)  # (dim/4,)

    # Apply sin/cos to x and y with different frequencies
    out_y = torch.einsum('hw,d->hwd', y, omega)  # (H, W, dim/4)
    out_x = torch.einsum('hw,d->hwd', x, omega)

    pos_y = torch.cat([out_y.sin(), out_y.cos()], dim=-1)
    pos_x = torch.cat([out_x.sin(), out_x.cos()], dim=-1)

    pos = torch.cat([pos_y, pos_x], dim=-1)  # (H, W, dim)

    if not for_NAT:
        # flatten
        pos = pos.reshape(crop_H * crop_W, embed_dim) # (H * W, dim)

    return pos


def config_get_activation(config, prefix="LectureNet.Network"):
    assert isinstance(config, Configuration)

    if config.contains(f"{prefix}.Activation"):
        activation_type = config.get(f"{prefix}.Activation.Type").lower()
        if activation_type == "gelu":
            return nn.GELU()
        elif activation_type == "silu":
            return nn.SiLU()
        elif activation_type == "relu":
            return nn.ReLU()
        elif activation_type == "leakyrelu":
            slope = config.get_float(f"{prefix}.Activation.Parameters.Slope")
            return nn.LeakyReLU(slope)
        elif activation_type == "sigmoid":
            return nn.Sigmoid()
        elif activation_type == "tanh":
            return nn.Tanh()
        else:
            raise Exception(f"Activation {activation_type} currently not supported!")
    else:
        # Default activation ....
        return nn.GELU()


class ConvNormalizationHelper:
    def __init__(self, norm_type, groups_type, groups_number, groups_size):
        self.__norm_type = norm_type.lower()
        self.__groups_type = groups_type
        self.__groups_number = groups_number
        self.__groups_size = groups_size

        if self.__norm_type not in ["batchnorm", "groupnorm"]:
            raise Exception(f"Unknown Normalization type: {norm_type}")

    def __repr__(self):
        if self.__norm_type == "batchnorm":
            return f"<BatchNorm2D, >"
        elif self.__norm_type == "groupnorm":
            if self.__groups_type.lower() == "fixednumber":
                sub_desc = f"by Fixed Number of Groups = {self.__groups_number}"
            elif self.__groups_type.lower() == "fixedsize":
                sub_desc = f"by Fixed Size per Group = {self.__groups_size}"
            else:
                sub_desc = "by UNKNOWN group type"

            return f"<GroupNorm: {sub_desc}>"
        else:
            return "<Unknown Normalization>"

    def get_layer(self, n_conv_out):
        if self.__norm_type == "batchnorm":
            return nn.BatchNorm2d(n_conv_out)
        elif self.__norm_type == "groupnorm":
            if self.__groups_type.lower() == "fixednumber":
                # all layers have the same fixed number of groups  (n_conv_out should be divisible by this number)
                num_group = self.__groups_number
            elif self.__groups_type.lower() == "fixedsize":
                # all layers will have a dynamic number of groups
                # so that the groups have a fixed size!
                if n_conv_out % self.__groups_size == 0:
                    num_group = n_conv_out // self.__groups_size
                else:
                    print(f"GroupNorm Warning: {n_conv_out} conv. maps are not divisible by {self.__groups_size}, defaulting to Groups=1")
                    num_group = 1
            else:
                raise Exception(f"Unknown Groups type: {self.__groups_type}. Only FixedNumber and FixedSize supported")

            return nn.GroupNorm(num_group, n_conv_out)
        else:
            raise Exception(f"Unknown Normalization type: {self.__norm_type}")

    @staticmethod
    def FromConfig(norm_config):
        assert isinstance(norm_config, Configuration)

        norm_type = norm_config.get("Type")
        groups_type = norm_config.get("GroupsType")
        groups_number = norm_config.get("GroupsNum", -1)
        groups_size = norm_config.get("GroupsSize", -1)

        return ConvNormalizationHelper(norm_type, groups_type, groups_number, groups_size)
