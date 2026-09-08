import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Tuple, Dict

from .util import get_2d_sincos_relative_positional_embed
from .WIN_layers import WindowBlock


class WindowFasterAttention2D(nn.Module):
    """
    Local (sliding) neighborhood attention for 2D token grids.
    This version uses a simplified Learnable RPB that avoids fully materializing the windows ..

    Input:  (B, H, W, C)  contiguous
    Output: (B, H, W, C)
    """

    def __init__(self, dim, num_heads, window_size, qkv_bias, proj_bias, attn_drop, proj_drop, use_rpb, mask_cache):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert window_size % 2 == 1, "window_size must be odd"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = window_size
        self.radius = window_size // 2  # padding?
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=proj_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rpb = use_rpb

        if self.use_rpb:
            # One bias per head per relative offset in the neighborhood (ky,kx) -> kernel_size^2 entries
            # Shape: (num_heads, K) where K=kernel_size^2
            self.rpb_table = nn.Parameter(torch.zeros(self.num_heads, self.kernel_size * self.kernel_size))
            nn.init.trunc_normal_(self.rpb_table, std=0.02)

        self.mask_cache = mask_cache


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C)
        """
        assert x.ndim == 4, "Expected (B,H,W,C)"
        B, H, W, C = x.shape
        assert C == self.dim, f"Expected C={self.dim}, got {C}"
        # assert x.is_contiguous(), "Input must be contiguous as (B,H,W,C)"

        # ---- QKV projections ----
        # (B, H, W, 3C)
        qkv = self.qkv(x)
        # each one is (B, H, W, C)
        q, k, v = qkv.chunk(3, dim=-1)

        N = H * W
        h = self.num_heads
        d = self.head_dim

        neighbor_idx, rpb_idx, valid_mask = self.mask_cache.get_metadata(H, W, x.device)
        K2 = neighbor_idx.shape[1]

        # (B, H, W, C) -> (B, N, h, d) -> (B, h, N, d)
        q = q.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()

        # flatten local neighbor indices
        flat_idx = neighbor_idx.reshape(-1)  # (N * K2,)

        # gather local K and V
        # result: (B, h, N*K2, d) -> (B, h, N, K2, d)
        k_local = k[:, :, flat_idx, :].view(B, h, N, K2, d)
        v_local = v[:, :, flat_idx, :].view(B, h, N, K2, d)

        if self.use_rpb:
            # local qk scores: (B, h, N, K2)
            scores = (q.unsqueeze(3) * k_local).sum(dim=-1) * self.scale

            # add learnable relative positional bias

            # self.rpb_table: (h, num_rel)
            # rpb_idx: (N, K2)
            bias = self.rpb_table[:, rpb_idx.reshape(-1)].view(h, N, K2)
            scores = scores + bias.unsqueeze(0)

            # mask invalid neighbors
            scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # softmax over neighborhood
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_drop(attn)

            # weighted sum of V: (B, h, N, d)
            out = (attn.unsqueeze(-1) * v_local).sum(dim=3)
        else:
            out = F.scaled_dot_product_attention(
                q.unsqueeze(-2),  # (B, h, N, 1, d)
                k_local,  # (B, h, N, K2, d)
                v_local,  # (B, h, N, K2, d)
            )

            out = out.squeeze(-2)  # (B, h, N, d)


        # merge heads back: (B, h, N, d) -> (B, H, W, C)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H, W, C)

        # convert to channels-first -> (B, C, H, W)
        out = out.permute(0, 3, 1, 2).contiguous()

        # 1x1 conv projection + dropout
        out = self.proj(out)
        out = self.proj_drop(out)

        # used when proj is linear
        # (B, HW, C) -> (B, H, W, C)
        # out = out.view(B, H, W, C) # .contiguous()

        # used when proj is a 1x1 conv
        # (B, C, H, W) -> (B, H, W, C)
        out = out.permute(0, 2, 3, 1)

        return out


class WindowFasterMaskCache(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        # Cache for border-valid masks for the last seen (H,W,device,dtype)
        self._mask_cache: Dict[Tuple[int, int, torch.device, torch.dtype, torch.Tensor]] = {}

    @torch.no_grad()
    def _build_metadata(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.kernel_size // 2

        # Query coordinates: (N,)
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        qy = yy.reshape(-1)  # (N,)
        qx = xx.reshape(-1)  # (N,)

        # Relative offsets: (K,)
        dys = torch.arange(-r, r + 1, device=device)
        dxs = torch.arange(-r, r + 1, device=device)
        dyy, dxx = torch.meshgrid(dys, dxs, indexing="ij")
        rel_dy = dyy.reshape(-1)  # (K,)
        rel_dx = dxx.reshape(-1)  # (K,)

        # Broadcast to all query positions: (N, K)
        ny = qy[:, None] + rel_dy[None, :]
        nx = qx[:, None] + rel_dx[None, :]

        # Validity mask: (N, K)
        valid_mask = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)

        # Flat neighbor indices, dummy 0 for invalid
        neighbor_idx = torch.zeros((H * W, self.kernel_size * self.kernel_size), dtype=torch.long, device=device)
        flat_valid_idx = ny[valid_mask] * W + nx[valid_mask]
        neighbor_idx[valid_mask] = flat_valid_idx.long()

        # RPB index for offsets only: (K,) -> broadcast to (N, K)
        # uses only the k*k offsets actually present in the local window
        rel_index = ((rel_dy + r) * self.kernel_size + (rel_dx + r)).long()  # (K,)
        rpb_idx = rel_index.unsqueeze(0).expand(H * W, -1).contiguous()

        return neighbor_idx, rpb_idx, valid_mask

    @torch.no_grad()
    def get_metadata(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = f"{H}x{W}"

        # check if first time seeing this combo of H and W
        if key not in self._mask_cache:
            print(f"... preparing new buffers for H={H}, W={W}")
            # create the mask ...
            neighbor_idx, rpb_idx, valid_mask = self._build_metadata(H, W, device=device)

            n_name = f"_neighbor_idx_{key}"
            r_name = f"_rpb_idx_{key}"
            v_name = f"_valid_mask_{key}"
            # register as non-persistent buffers ....
            self.register_buffer(n_name, neighbor_idx, persistent=False)
            self.register_buffer(r_name, rpb_idx, persistent=False)
            self.register_buffer(v_name, valid_mask, persistent=False)

            # store in cache
            self._mask_cache[key] = (n_name, r_name, v_name)
            print("...Buffers created!....")

        # get combo from cache ...
        n_name, r_name, v_name = self._mask_cache[key]

        neighbor_idx = getattr(self, n_name)
        rpb_idx = getattr(self, r_name)
        valid_mask = getattr(self, v_name)

        # safety: if metadata was created on another device, move lazily
        if neighbor_idx.device != device:
            neighbor_idx = neighbor_idx.to(device)
            rpb_idx = rpb_idx.to(device)
            valid_mask = valid_mask.to(device)

            setattr(self, n_name, neighbor_idx)
            setattr(self, r_name, rpb_idx)
            setattr(self, v_name, valid_mask)

        return neighbor_idx, rpb_idx, valid_mask


class WindowFasterBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, dropout, activation, qkv_bias, proj_bias,
                 use_rel_pos_bias, mask_cache, use_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint  #

        # LayerNorm before attention (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)

        # Placeholder for NAT
        self.win_attn = WindowFasterAttention2D(dim, num_heads, window_size, qkv_bias, proj_bias, dropout, dropout,
                                                use_rel_pos_bias, mask_cache)

        # TODO: will this be needed?
        self.drop_path = nn.Identity()  # optional stochastic depth

        self.norm2 = nn.LayerNorm(dim)

        # follows the "feed forward block" of transformer encoder layer
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (N, C, H, W)
        returns: same shape
        """
        N, C, H, W = x.shape
        shortcut = x
        # from (N, C, H, W) to (N, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        # then, normalize (norm-fist style)
        x = self.norm1(x)

        # For Neighborhood Attention (NAT-style):
        # compute the attention at once
        # x_attn = self.win_attn(x)
        if self.training and self.use_checkpoint:  # gate it; don't checkpoint in eval
            x_attn = checkpoint(self.win_attn, x, use_reentrant=False)
        else:
            x_attn = self.win_attn(x)

        # from (H, H, W, C) to (N, C, H, W)
        x = x_attn.permute(0, 3, 1, 2).contiguous()

        # Residual connection
        x = shortcut + self.drop_path(x)

        # MLP block
        shortcut2 = x
        # from (N, C, H, W) back to (N, H, W, C)
        x = self.norm2(x.permute(0, 2, 3, 1).contiguous())
        if self.training and self.use_checkpoint:
            x = checkpoint(self.mlp, x, use_reentrant=False)
        else:
            x = self.mlp(x)
        # from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = shortcut2 + self.drop_path(x)

        return x


class WINDTransBlock(nn.Module):
    def __init__(self, faster, latent_dim, n_layers, n_heads, dff_ratio, drop_out, layer_activation, post_layer_norm,
                 mask_neighbors, qkv_bias, proj_bias, use_rel_pos_bias):
        super(WINDTransBlock, self).__init__()

        self.__latent_dim = latent_dim
        self.__n_layers = n_layers
        self.__n_heads = n_heads
        self.__dff_ratio = dff_ratio
        self.__drop_out = drop_out
        self.__layer_activation = layer_activation
        self.__use_rel_pos_bias = use_rel_pos_bias
        # this works like regular window (1=1x1, 3=3x3, etc.)
        self.__mask_neighbors = mask_neighbors
        self.__qkv_bias = qkv_bias
        self.__proj_bias = proj_bias

        if n_layers == 0:
            # bypass mode
            self.__trans_encoder = nn.Identity()
        else:
            if faster:
                # use newer block that avoids unfold ...
                mask_cache = WindowFasterMaskCache(mask_neighbors)
                layer_list = [
                    WindowFasterBlock(latent_dim, n_heads, mask_neighbors, dff_ratio, drop_out, layer_activation,
                                      qkv_bias, proj_bias, use_rel_pos_bias, mask_cache,  False)  # ((i + 2) % 3) < 2
                    for i in range(n_layers)
                ]
            else:
                # use earlier block that uses unfold ...
                layer_list = [
                    WindowBlock(latent_dim, n_heads, mask_neighbors, dff_ratio, drop_out,layer_activation, qkv_bias,
                                proj_bias, use_rel_pos_bias, False)  # ((i + 2) % 3) < 2
                    for i in range(n_layers)
                ]
            self.__trans_encoder = nn.Sequential(*layer_list)

        if post_layer_norm:
            self.__post_norm = nn.LayerNorm(latent_dim)
        else:
            self.__post_norm = nn.Identity()

        # initializing weights
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, conv_feats):
        # N = batch size, maps_h = num. of height patches, maps_w = num. of width patches
        N, _, maps_h, maps_w = conv_feats.shape

        if self.__use_rel_pos_bias:
            # use the input as given ..., the Window Block will use the internal relative positional biases
            tempo = conv_feats
        else:
            # using global encoding ..
            if self.training:
                # make it like look like a random crop from random size during training
                r = torch.rand(4, device=conv_feats.device)
                img_h = maps_h + torch.round(r[0] * maps_h * 9).to(torch.long)
                img_w = maps_w + torch.round(r[1] * maps_w * 9).to(torch.long)
                offset_y = torch.floor(r[2] * (img_h + 1 - maps_h)).to(torch.long)
                offset_x = torch.floor(r[3] * (img_w + 1 - maps_w)).to(torch.long)
            else:
                # no offset added, treat it as crop of the size of the entire image ...
                offset_x = torch.tensor(0, device=conv_feats.device, dtype=torch.long)
                offset_y = torch.tensor(0, device=conv_feats.device, dtype=torch.long)
                img_h, img_w = maps_h, maps_w

            # compute the same positional encoding used for windowed attention
            # (H, W, dim)
            pos_embedding = get_2d_sincos_relative_positional_embed(self.__latent_dim, offset_y, offset_x, maps_h, maps_w,
                                                                    img_h, img_w, for_NAT=True)

            pos_embedding = pos_embedding.to(conv_feats.device)
            # (H, W, dim) -> (dim, H, W)
            pos_embedding = pos_embedding.permute(2, 0, 1)

            # (N, dim, H, W)
            tempo = conv_feats + pos_embedding

        # TODO: this is inefficient, maybe I can change the layer to expect the original format ...
        # transpose to expected format .... (N, H, W, dim)
        # tempo = tempo.permute(0, 2, 3, 1).contiguous()
        # input("after permute, before trans-encoder")
        # .... apply encoder ...
        enc_output = self.__trans_encoder(tempo)
        # input("after trans-encoder, before permutting the outputs")

        # and back to the original format
        # (N, H, W, dim) -> (N, dim, H, W)
        # enc_output = enc_output.permute(0, 3, 1, 2).contiguous()

        if isinstance(self.__post_norm, nn.LayerNorm):
            # must change the format of the input

            # N x Latent x MapH x MapW -> N x Latent x T
            img_seq_len = maps_h * maps_w
            enc_output = enc_output.reshape(N, self.__latent_dim, img_seq_len)

            # N x Latent x T  -> N x T x Latent
            enc_output = enc_output.transpose(1, 2)

            # apply post normalization
            enc_output = self.__post_norm(enc_output)

            # now ... move it back to conv feature map style ...
            # N x T x Latent ->  N x Latent x T
            enc_output = enc_output.transpose(1, 2)

            # the second param here is the "size" of each slice (size of a row),
            # and the third is the "step" (jump to the next slice)
            enc_output = enc_output.unfold(2, maps_w, maps_w)

        return enc_output

