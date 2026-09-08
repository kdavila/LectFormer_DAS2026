
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Optional, Tuple, Dict


class WindowAttention2D(nn.Module):
    """
    Local (sliding) neighborhood attention for 2D token grids.

    Input:  (B, H, W, C)  contiguous
    Output: (B, H, W, C)

    Complexity: O(B * H * W * kernel_size^2)  (no global N^2).
    """

    def __init__(self, dim, num_heads, window_size, qkv_bias, proj_bias, attn_drop, proj_drop, use_rpb):
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
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.kernel_size * self.kernel_size))
            nn.init.trunc_normal_(self.rpb, std=0.02)

        # Cache for border-valid masks for the last seen (H,W,device,dtype)
        self._mask_cache: Dict[Tuple[int, int, torch.device, torch.dtype, torch.Tensor]] = {}

    @torch.no_grad()
    def _get_valid_neighbor_mask(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns additive mask for invalid neighbors due to borders.
        Shape: (1, 1, HW, K) where K=kernel_size^2
        Values: 0 for valid neighbors, -inf for invalid.
        """
        key = f"{H}x{W}x{device}x{dtype}"
        if key in self._mask_cache:
            return self._mask_cache[key]

        K = self.kernel_size * self.kernel_size

        # query coordinates
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
        qy = yy.reshape(-1)  # (HW,)
        qx = xx.reshape(-1)  # (HW,)

        # neighbor offsets in a fixed order matching unfold's patch order:
        # unfold uses a raster order over the kernel: top-left -> bottom-right
        off_y = torch.arange(-self.radius, self.radius + 1, device=device)
        off_x = torch.arange(-self.radius, self.radius + 1, device=device)
        oy, ox = torch.meshgrid(off_y, off_x, indexing="ij")  # (ks,ks)
        oy = oy.reshape(-1)  # (K,)
        ox = ox.reshape(-1)  # (K,)

        ny = qy[:, None] + oy[None, :]  # (HW,K)
        nx = qx[:, None] + ox[None, :]  # (HW,K)

        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)  # (HW,K)

        # additive mask: 0 for valid, -inf for invalid
        mask = torch.zeros((H * W, K), device=device, dtype=dtype)
        mask = mask.masked_fill(~valid, float("-inf"))
        mask = mask.view(1, 1, H * W, K)

        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C)
        """
        assert x.ndim == 4, "Expected (B,H,W,C)"
        B, H, W, C = x.shape
        assert C == self.dim, f"Expected C={self.dim}, got {C}"
        # assert x.is_contiguous(), "Input must be contiguous as (B,H,W,C)"

        HW = H * W
        # ks = self.kernel_size
        K = self.kernel_size * self.kernel_size
        # r = self.radius

        # ---- QKV projections ----
        # (B, H, W, 3C)
        qkv = self.qkv(x)
        # each one is (B, H, W, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for heads
        # q: (B, heads, HW, head_dim)
        q = q.view(B, HW, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Gather neighborhood K/V with unfold ----
        # unfold expects (B, C, H, W). We need per-head gathering, so flatten heads into channels.
        # (B, H, W, C) -> (B, C, H, W)
        # k_2d: (B, heads*head_dim, H, W)
        k_2d = k.permute(0, 3, 1, 2) # .contiguous()  # (B, C, H, W)
        v_2d = v.permute(0, 3, 1, 2) # .contiguous()  # (B, C, H, W)

        # add padding (expensive, requires materializing)
        k_pad = F.pad(k_2d, (self.radius, self.radius, self.radius, self.radius))
        v_pad = F.pad(v_2d, (self.radius, self.radius, self.radius, self.radius))
        # then, crete the unfolded views ...
        k_nb = k_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        v_nb = v_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # and reshape ...
        # (B, C, H, W, ksize, ksize) -> (B, C, H, W, K)
        k_nb = k_nb.reshape(B, C, H, W, K)
        v_nb = v_nb.reshape(B, C, H, W, K)
        # then split heads ...
        # (B, C, H, W, K) -> (B, heads, head_dim, H, W, K)
        k_nb = k_nb.view(B, self.num_heads, self.head_dim, H, W, K)
        v_nb = v_nb.view(B, self.num_heads, self.head_dim, H, W, K)
        # finally move for attention
        # (B, heads, head_dim, H, W, K) -> (B, heads, H, W, K, head_dim) -> (B, heads, HW, K, head_dim)
        k_nb = k_nb.permute(0, 1, 3, 4, 5, 2).reshape(B, self.num_heads, H * W, K, self.head_dim)
        v_nb = v_nb.permute(0, 1, 3, 4, 5, 2).reshape(B, self.num_heads, H * W, K, self.head_dim)

        # ---- Attention computation (optionally chunked over query positions) ----
        # q: (B, heads, HW, head_dim)
        # k_nb: (B, heads, HW, K, head_dim)
        # logits: (B, heads, HW, K)
        valid_mask = self._get_valid_neighbor_mask(H, W, device=x.device, dtype=x.dtype)

        if self.use_rpb:
            # (1, heads, 1, K) broadcast over B and HW
            rpb = self.rpb.view(1, self.num_heads, 1, K).to(dtype=x.dtype, device=x.device)
        else:
            rpb = None

        if rpb is not None:
            valid_mask = valid_mask + rpb

        q_sdpa = q.permute(0, 2, 1, 3).unsqueeze(-2)  # (B, HW, heads, 1, d)
        k_sdpa = k_nb.permute(0, 2, 1, 3, 4)  # (B, HW, heads, K, d)
        v_sdpa = v_nb.permute(0, 2, 1, 3, 4)  # (B, HW, heads, K, d)

        mask_sdpa = valid_mask.permute(0, 2, 1, 3).unsqueeze(-2)  # (1, HW, 1, 1, K)

        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=mask_sdpa,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,  # optional if you already define your own scale
        )  # (B, HW, heads, 1, d)

        out = out.squeeze(-2).permute(0, 2, 1, 3)  # (B, heads, HW, d)

        # ---- Merge heads and project ----
        # using a 1x1 convolutional layer
        # (B, heads, HW, d) -> (B, HW, heads, d)
        out = out.permute(0, 2, 1, 3)
        # (B, HW, heads, d) -> (B, HW, C)
        out = out.reshape(B, HW, C)
        # (B, HW, C) -> (B, C, HW)
        out = out.transpose(1, 2)
        # (B, C, HW) -> (B, C, H, W)
        out = out.reshape(B, C, H, W)
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


class WindowBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, dropout, activation, qkv_bias, proj_bias,
                 use_rel_pos_bias, use_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint #

        # LayerNorm before attention (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)

        # Placeholder for NAT
        self.win_attn = WindowAttention2D(
            dim, num_heads, window_size, qkv_bias, proj_bias, dropout, dropout, use_rel_pos_bias
        )

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

    @torch._dynamo.disable
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

        # this is the transformation for the original flattened multi-head attention
        # x = x_attn.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        # for NAT
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

