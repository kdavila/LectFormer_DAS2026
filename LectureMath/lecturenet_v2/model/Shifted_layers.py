
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import get_2d_sincos_relative_positional_embed, compile_disable_if_windows


class ShiftedWindowAttention2D(nn.Module):
    """
    Non-overlapping 2D window attention on BHWC feature maps, with:
      - optional learnable relative positional bias (RPB)
      - arbitrary window shift (px, py) passed at forward time
      - padding + masking for invalid tokens caused by shifted partitioning

    Input:
        x:  (B, H, W, C)

    Output:
        y:  (B, H, W, C)

    """
    def __init__(self,
        dim: int,
        num_heads: int,
        window_size,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rpb: bool = True
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.wh, self.ww = window_size
        self.attn_drop = attn_drop

        self.use_rpb = use_rpb

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # ------------------------------------------------------------
        # Relative positional bias (Swin-style)
        # ------------------------------------------------------------
        # Table size: (2*wh - 1) * (2*ww - 1), one bias value per head
        if use_rpb:
            num_relative_positions = (2 * self.wh - 1) * (2 * self.ww - 1)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(num_relative_positions, num_heads)
            )

            # Precompute pairwise relative position index for one window
            relative_position_index = self._build_relative_position_index(self.wh, self.ww)
            self.register_buffer("relative_position_index", relative_position_index, persistent=False)

            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.register_buffer("relative_position_index", None, persistent=False)
            self.relative_position_bias_table = None

    @staticmethod
    def _build_relative_position_index(wh: int, ww: int) -> torch.Tensor:
        """
        Returns:
            relative_position_index: (T, T), where T = wh * ww
        """
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, wh, ww)
        coords_flatten = coords.reshape(2, -1)  # (2, T)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, T, T)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (T, T, 2)

        relative_coords[:, :, 0] += wh - 1
        relative_coords[:, :, 1] += ww - 1
        relative_coords[:, :, 0] *= (2 * ww - 1)

        relative_position_index = relative_coords.sum(-1)  # (T, T)
        # indices are a TxT matrix, where right-top position is 0
        # then, numbers grow in diagonals... all the way to the left-bottom
        # [[2 1 0]
        #  [3 2 1]
        #  [4 3 2]]

        return relative_position_index

    def _get_relative_position_bias(self, device, dtype):
        """
        Returns:
            bias: (1, num_heads, 1, T, T)
        """
        T = self.wh * self.ww

        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]  # (T*T, num_heads)

        bias = bias.view(T, T, self.num_heads)          # (T, T, h)
        bias = bias.permute(2, 0, 1).contiguous()       # (h, T, T)
        bias = bias.unsqueeze(0).unsqueeze(2)           # (1, h, 1, T, T)
        return bias.to(device=device, dtype=dtype)

    @staticmethod
    def _pad_bhwc(x: torch.Tensor, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int):
        """
        Pad BHWC tensor spatially.
        """
        # Convert BHWC -> BCHW for F.pad
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def _window_partition(self, x: torch.Tensor):
        """
        x: (B, Hp, Wp, C_or_d)
        returns: (B, num_windows, T, C_or_d)
        """
        B, Hp, Wp, C = x.shape
        # wh, ww = ,

        # compute the resulting hor/ver windows
        # (this assumes that the map has been padded to fit an exact number of windows)
        nWh = Hp // self.wh
        nWw = Wp // self.ww
        T = self.wh * self.ww

        # generate the windows by splitting into hor/ver groups ...
        x = x.view(B, nWh, self.wh, nWw, self.ww, C)
        # change the order of dimensions to get window index first, then patch per window indices
        # then make sure it is contiguous.
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()    # (B, nWh, nWw, wh, ww, C)
        # and then flatten the window index (from 2D to 1 dim=nW)
        # and also flatten each window from wh * ww to T.
        x = x.view(B, nWh * nWw, T, C)                  # (B, nW, T, C)
        return x

    def _window_reverse(self, x: torch.Tensor, Hp: int, Wp: int):
        """
        x: (B, num_windows, T, C)
        returns: (B, Hp, Wp, C)
        """
        B, num_windows, T, C = x.shape
        # this assumes the windows are divisible
        nWh = Hp // self.wh
        nWw = Wp // self.ww

        # from flattened windows and per-window tokens to array of
        # 2D windows with 2D map of tokens
        x = x.view(B, nWh, nWw, self.wh, self.ww, C)
        # change back to the original order and make contiguous
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()    # (B, nWh, wh, nWw, ww, C)
        # reshape back to leave the dims last.
        x = x.view(B, Hp, Wp, C)
        return x

    @torch.compiler.disable
    def eager_block(self, q, k, v, valid, B, C, H, W, Hp, Wp, pad_top, pad_left):
        # ------------------------------------------------------------
        # Partition into windows
        # ------------------------------------------------------------
        q = self._window_partition(q)  # (B, nW, T, C)
        k = self._window_partition(k)  # (B, nW, T, C)
        v = self._window_partition(v)  # (B, nW, T, C)
        valid_w = self._window_partition(valid)  # (1, nW, T, 1)

        nW = q.shape[1]
        T = q.shape[2]

        # Split heads
        # (B, nW, T, C) -> (B, nW, T, h, d) -> (B, h, nW, T, d)
        q = q.view(B, nW, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        k = k.view(B, nW, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        v = v.view(B, nW, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
        # q,k,v: (B, h, nW, T, d)

        # ------------------------------------------------------------
        # Attention mask
        # ------------------------------------------------------------
        # Key-valid mask only. Query-valid rows are zeroed after attention.
        # Shape: (1, 1, nW, 1, T), additive mask with -inf on invalid keys.
        key_valid = valid_w.squeeze(-1).unsqueeze(1).unsqueeze(3)  # (1, 1, nW, 1, T)

        # find the minimum for current data type
        neg_inf = torch.finfo(q.dtype).min
        # these () lead to scalar values, either:
        #   0.0 where valid or
        #   -inf where invalid.
        key_mask = torch.where(
            key_valid > 0,
            torch.zeros((), device=q.device, dtype=q.dtype),
            torch.full((), neg_inf, device=q.device, dtype=q.dtype),
        )

        attn_mask = key_mask  # start from validity mask

        if self.use_rpb:
            rpb = self._get_relative_position_bias(q.device, q.dtype)  # (1, h, 1, T, T)
            # will this work???
            attn_mask = attn_mask + rpb

        # ------------------------------------------------------------
        # SDPA
        # ------------------------------------------------------------
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (B, h, nW, T, d)

        # Zero out invalid query outputs
        query_valid = valid_w.squeeze(-1).unsqueeze(1).unsqueeze(-1)  # (1, 1, nW, T, 1)
        out = out * query_valid

        return out, nW, T

    def forward(self, x: torch.Tensor, px: int = 0, py: int = 0):
        """
        x:  (B, H, W, C)
        px: horizontal shift of window grid
        py: vertical shift of window grid

        Typical useful range:
            0 <= px < window_width
            0 <= py < window_height

        Larger integers are reduced modulo the window size.
        """
        B, H, W, C = x.shape

        # Periodic equivalence for shifted window partition
        px = int(px) % self.ww
        py = int(py) % self.wh

        # ------------------------------------------------------------
        # Shifted partition by padding on top/left first
        # Then add right/bottom padding so spatial size is divisible
        # by window size.
        # ------------------------------------------------------------
        # Take these as "negative" coordinates
        # * px shift window start to the "left"
        # * py shift window start above the "top"
        pad_left = px
        pad_top = py

        Hp_pre = H + pad_top
        Wp_pre = W + pad_left

        pad_bottom = (self.wh - (Hp_pre % self.wh)) % self.wh
        pad_right = (self.ww - (Wp_pre % self.ww)) % self.ww

        Hp = H + pad_top + pad_bottom
        Wp = W + pad_left + pad_right

        # Spatially pad the feature map
        x_pad = self._pad_bhwc(x, pad_left, pad_right, pad_top, pad_bottom)  # (B, Hp, Wp, C)

        # Valid-token map: 1 for real tokens, 0 for padded tokens
        valid = x.new_ones((1, H, W, 1))
        valid = self._pad_bhwc(valid, pad_left, pad_right, pad_top, pad_bottom)  # (1, Hp, Wp, 1)

        # ------------------------------------------------------------
        # QKV
        # ------------------------------------------------------------
        qkv = self.qkv(x_pad)  # (B, Hp, Wp, 3C)

        qkv = qkv.view(B, Hp, Wp, 3, self.dim)
        qkv = qkv.permute(3, 0, 1, 2, 4)  # (3, B, Hp, Wp, C)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, Hp, Wp, C)

        q = q * valid
        k = k * valid
        v = v * valid

        out, nW, T = self.eager_block(q, k, v, valid, B, C, H, W, Hp, Wp, pad_top, pad_left)

        # ------------------------------------------------------------
        # Merge windows back
        # ------------------------------------------------------------
        # (B, h, nW, T, d) -> (B, nW, T, h, d)
        out = out.permute(0, 2, 3, 1, 4).contiguous()  # (B, nW, T, h, d)
        # (B, nW, T, h, d) -> (B, nW, T, C)
        out = out.view(B, nW, T, C)
        # (B, nW, T, C) -> (B, Hp, Wp, C)
        out = self._window_reverse(out, Hp, Wp)  # (B, Hp, Wp, C)

        # Remove padding
        out = out[:, pad_top:pad_top + H, pad_left:pad_left + W, :]  # (B, H, W, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class ShiftedBlock(nn.Module):
    TypeNormal = 0
    TypeShifted = 1
    TypeHybrid = 2

    def __init__(self, dim, num_heads, window_size, mlp_ratio, dropout, activation, qkv_bias, proj_bias,
                 use_rel_pos_bias, block_type):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.block_type = block_type

        # LayerNorm before attention (pre-norm style)
        self.norm1 = nn.LayerNorm(dim)

        self.win_attn = ShiftedWindowAttention2D(dim, num_heads, window_size, qkv_bias, proj_bias, dropout, dropout,
                                                 use_rel_pos_bias)

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

    def exec_transformer(self, x):
        if self.block_type in [ShiftedBlock.TypeNormal, ShiftedBlock.TypeHybrid]:
            # compute attention on normal windows ...
            x_normal = self.win_attn(x, 0, 0)
        else:
            x_normal = None

        if self.block_type in [ShiftedBlock.TypeShifted, ShiftedBlock.TypeHybrid]:
            # compute attention on shifted windows ...
            # shift by half of the window size
            r = self.window_size // 2
            x_shifted = self.win_attn(x, r, r)
        else:
            x_shifted = None

        if self.block_type == ShiftedBlock.TypeNormal:
            return x_normal
        elif self.block_type == ShiftedBlock.TypeShifted:
            return x_shifted
        else:
            # combine and re-scale so that it will not overpower the residual
            return (x_normal + x_shifted) * 0.5

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

        x_attn = self.exec_transformer(x)

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


class ShiftedTransBlock(nn.Module):
    def __init__(self, latent_dim, n_layers, n_heads, dff_ratio, drop_out, layer_activation, post_layer_norm,
                 mask_neighbors, qkv_bias, proj_bias, use_rel_pos_bias, use_hybrid_layers):
        super(ShiftedTransBlock, self).__init__()

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
        self.__use_hybrid_layers = use_hybrid_layers

        if n_layers == 0:
            # bypass mode
            self.__trans_encoder = nn.Identity()
        else:
            layer_list = []
            for i in range(n_layers):
                if use_hybrid_layers:
                    # all layers will be hybrid ...
                    block_type = ShiftedBlock.TypeHybrid
                else:
                    # alternate between normal and shifted windows ...
                    if i % 2 == 0:
                        block_type = ShiftedBlock.TypeNormal
                    else:
                        block_type = ShiftedBlock.TypeShifted

                layer_list.append(
                    ShiftedBlock(latent_dim, n_heads, mask_neighbors, dff_ratio, drop_out, layer_activation,
                                 qkv_bias, proj_bias, use_rel_pos_bias, block_type)
                )

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

    @compile_disable_if_windows
    def eager_transformer(self, x):
        return self.__trans_encoder(x)

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

        # .... apply encoder ...
        enc_output = self.__trans_encoder(tempo)
        # enc_output = self.eager_transformer(tempo)

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

