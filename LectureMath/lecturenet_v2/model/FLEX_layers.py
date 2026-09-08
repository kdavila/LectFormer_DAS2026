
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from .util import get_2d_sincos_relative_positional_embed

compiled_flex = torch.compile(flex_attention, dynamic=True)


class Local2DFlexAttention(nn.Module):
    """
    Local 2D attention with shared relative positional bias (RPB).

    Input:  x of shape [B, H, W, C]
    Output: y of shape [B, H, W, C]

    This is a "sliding local attention" formulation:
      - each token attends only to tokens within a local 2D window
      - relative positional bias depends only on (dy, dx), shared globally

    It is NOT the same as partitioning into disjoint windows,
    nor the same as a full RPB where local (dy, dx) matter as a function of the relative position to window center
    """

    def __init__(self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        use_rpb: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert window_size % 2 == 1

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.radius = window_size // 2
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # projection dropout ... easy to implement
        self.proj_drop = nn.Dropout(proj_drop)
        # attention dropout ... not directly available apparently ...?

        self.use_rpb = use_rpb
        if self.use_rpb:
            """
            # Note that these work but are TOO slow!! slower than the other WIND implementation
            # Shared relative positional bias:
            # [num_heads, 2r+1, 2r+1]
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, 2 * self.radius + 1, 2 * self.radius + 1)
            )
            """
            raise NotImplementedError("RPB was not implemented for the FlexAttention backend")

        # Cache BlockMask per (H, W, device)
        self._block_mask_cache: Dict[Tuple[int, int, str], object] = {}

    def _make_mask_mod(self, W: int):
        r = self.radius

        def mask_mod(_batch, _head, q_idx, kv_idx):
            # q_idx, kv_idx are flattened spatial indices in [0, H*W)
            qy = q_idx // W
            qx = q_idx % W
            ky = kv_idx // W
            kx = kv_idx % W

            dy = ky - qy
            dx = kx - qx

            return (torch.abs(dy) <= r) & (torch.abs(dx) <= r)

        return mask_mod

    def _make_score_mod(self, W: int):
        # rpb = self.rpb
        r = self.radius
        num_heads_f = float(self.num_heads - 1) if self.num_heads > 1 else 1.0

        def score_mod(score, _batch, head, q_idx, kv_idx):
            # from 1D sequence indices to 2D positions
            qy = q_idx // W
            qx = q_idx % W
            ky = kv_idx // W
            kx = kv_idx % W
            # relative distances ....
            dy = ky - qy
            dx = kx - qx
            # check the valid pairs (within the same window)
            valid = (torch.abs(dy) <= r) & (torch.abs(dx) <= r)
            # and get valid distances
            safe_dy = torch.where(valid, dy, torch.zeros_like(dy))
            safe_dx = torch.where(valid, dx, torch.zeros_like(dx))
            # 44.s

            # Learnable bias: Slow and error prone!
            # bias = rpb[head, safe_dy + r, safe_dx + r]

            # using a Fixed Relative Positional Bias, based on a quadratic function
            # elements near the center get more attention, while the neighbors get less attention

            # the total penalty changes dynamically per head
            # per-head weight ... some heads get lower alpha, others get higher alpha ...
            head_f = head.to(score.dtype)
            head_norm = head_f / num_heads_f
            alpha_min = 0.5
            alpha_max = 2.0
            alpha = alpha_min + head_norm * (alpha_max - alpha_min)

            # now compute the distances from center ...
            safe_dy_f = safe_dy.to(score.dtype)
            safe_dx_f = safe_dx.to(score.dtype)
            dist2 = safe_dy_f * safe_dy_f + safe_dx_f * safe_dx_f
            #  re-weight raw distances by max distance, and then multiply by alpha
            bias = -alpha * (dist2 / float(2 * r * r))

            return score + bias
            # return score + torch.where(valid, bias, torch.zeros_like(score))

        return score_mod

    def _get_block_mask(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._block_mask_cache:
            N = H * W
            # mask_mod = self._make_mask_mod(H, W)
            mask_mod = self._make_mask_mod(W)

            # Since sparsity pattern is shared across batch and heads,
            # use B=None, H=None for broadcast.
            block_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=N,
                KV_LEN=N,
                device=device,
                _compile=True, # true
            )
            self._block_mask_cache[key] = block_mask

        return self._block_mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert C == self.dim

        N = H * W

        # [B, N, C]
        x_flat = x.reshape(B, N, C)

        # compute the Q, K, and V
        qkv = self.qkv(x_flat)
        # split: [B, N, 3C] -> 3 vectors of [B, N, C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, N, C] -> [B, N, heads, head_dim] -> [B, heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        block_mask = self._get_block_mask(H, W, x.device)
        score_mod = self._make_score_mod(W) if self.use_rpb else None

        # y = flex_attention(
        y = compiled_flex(
            q, k, v,
            block_mask=block_mask,
            score_mod=score_mod,
            scale=self.scale,
        )  # [B, heads, N, head_dim]

        # [B, heads, N, head_dim] -> [B, N, heads, head_dim] -> [B, N, C]
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)

        # [B, N, C] -> [B, H, W, C]
        y = y.view(B, H, W, C)
        return y


class FlexBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, dropout, activation, qkv_bias, proj_bias, use_rpb):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # LayerNorm before attention (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)

        # Placeholder for NAT
        self.flex_attn = Local2DFlexAttention(dim, num_heads, window_size, qkv_bias, proj_bias, dropout, use_rpb)

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

        # compute the attention
        x_attn = self.flex_attn(x)

        # from (N, H, W, C) to (N, C, H, W)
        x = x_attn.permute(0, 3, 1, 2).contiguous()

        # Residual connection
        x = shortcut + self.drop_path(x)

        # MLP block
        shortcut2 = x
        # from (N, C, H, W) back to (N, H, W, C)
        x = self.norm2(x.permute(0, 2, 3, 1).contiguous())
        x = self.mlp(x)
        # from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = shortcut2 + self.drop_path(x)

        return x


class FlexTransBlock(nn.Module):
    def __init__(self, latent_dim, n_layers, n_heads, dff_ratio, drop_out, layer_activation, post_layer_norm,
                 mask_neighbors, qkv_bias, proj_bias, use_rel_pos_bias):
        super(FlexTransBlock, self).__init__()

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
            layer_list = [
                FlexBlock(latent_dim, n_heads, mask_neighbors, dff_ratio, drop_out, layer_activation, qkv_bias,
                          proj_bias, use_rel_pos_bias)
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

        # input("after permute, before trans-encoder")
        # .... apply encoder ...
        enc_output = self.__trans_encoder(tempo)
        # input("after trans-encoder, before permutting the outputs")

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

