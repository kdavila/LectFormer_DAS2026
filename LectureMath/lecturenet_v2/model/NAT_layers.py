
import torch
import torch.nn as nn
import torch.nn.functional as F

import natten
from natten import NeighborhoodAttention2D
from .util import get_2d_sincos_relative_positional_embed


class NATBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio, qkv_bias, dropout, activation, use_rel_pos_bias):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        head_dim = dim // num_heads
        if head_dim & (head_dim - 1) != 0:
            print(f"\nWarning: With dim={dim} and num_heads={num_heads}, head_dim={head_dim} which is not a power of 2")
            print("          This configuration is not supported by NATTEN backends such as FlexAttention")

        # LayerNorm before attention (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)

        # Placeholder for NAT
        # self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        if natten.__version__.startswith("0.17"):
            self.nat_attn = NeighborhoodAttention2D(dim, num_heads=num_heads, kernel_size=window_size, dilation=1,
                                                    proj_drop=dropout, qkv_bias=qkv_bias, rel_pos_bias=use_rel_pos_bias)
        else:
            # newer versions?
            if use_rel_pos_bias:
                print("WARNING: RBP not available for this version of NATTEN!")
            self.nat_attn = NeighborhoodAttention2D(dim, num_heads=num_heads, kernel_size=window_size, dilation=1,
                                                    proj_drop=dropout, qkv_bias=qkv_bias)

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
    def eager_attention_block(self, x):
        return self.nat_attn(x)

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

        # for MultiheadAttention ...
        # Flatten spatial dimensions
        # x_flat = x.view(N, H * W, C)
        # computes global attention (inefficient)
        # x_attn, _ = self.attn(x_flat, x_flat, x_flat)

        # For Neighborhood Attention (NAT):
        # compute required padding (only if smaller than window)
        pad_h = max(0, self.window_size - H)
        pad_w = max(0, self.window_size - W)

        if pad_h > 0 or pad_w > 0:
            # F.pad expects (N, C, H, W), so permute
            x_perm = x.permute(0, 3, 1, 2)  # (N, C, H, W)

            # pad format: (left, right, top, bottom)
            # we pad only on the "end" to keep indexing simple
            x_pad = F.pad(x_perm, (0, pad_w, 0, pad_h),  mode="constant")

            # back to (N, H, W, C)
            x = x_pad.permute(0, 2, 3, 1).contiguous()

        # x_attn = self.nat_attn(x)
        x_attn = self.eager_attention_block(x)

        # --- crop back if padded ---
        if pad_h > 0 or pad_w > 0:
            x_attn = x_attn[:, :H, :W, :]

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
        x = self.mlp(x)
        # from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = shortcut2 + self.drop_path(x)

        return x


class NATTransBlock(nn.Module):
    def __init__(self, latent_dim, n_layers, n_heads, dff_ratio, qkv_bias, drop_out, layer_activation, post_layer_norm,
                 mask_neighbors, use_rel_pos_bias):
        super(NATTransBlock, self).__init__()

        self.__latent_dim = latent_dim
        self.__n_layers = n_layers
        self.__n_heads = n_heads
        self.__dff_ratio = dff_ratio
        self.__drop_out = drop_out
        self.__layer_activation = layer_activation
        self.__use_rel_pos_bias = use_rel_pos_bias
        # this works like regular window (1=1x1, 3=3x3, etc.)
        self._mask_neighbors = mask_neighbors

        if n_layers == 0:
            # bypass mode
            self.__trans_encoder = nn.Identity()
        else:
            layer_list = [
                NATBlock(latent_dim, n_heads, mask_neighbors, dff_ratio, qkv_bias, drop_out, layer_activation,
                         self.__use_rel_pos_bias)
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

        # pos_embedding = get_sinusoid_encoding(img_seq_len, d_hid=self.__latent_dim)
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

        if self.__use_rel_pos_bias:
            # use the input as given ..., the NAT Block will use the internal positional biases from NAT
            tempo = conv_feats
        else:
            # compute the same positional encoding used for windowed attention
            # (H, W, dim)
            pos_embedding = get_2d_sincos_relative_positional_embed(self.__latent_dim, offset_y, offset_x, maps_h, maps_w,
                                                                    img_h, img_w, for_NAT=True)

            pos_embedding = pos_embedding.to(conv_feats.device)
            # (H, W, dim) -> (dim, H, W)
            pos_embedding = pos_embedding.permute(2, 0, 1)

            # (N, dim, H, W)
            tempo = conv_feats + pos_embedding

        # .... apply encoder ...
        enc_output = self.__trans_encoder(tempo)

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

