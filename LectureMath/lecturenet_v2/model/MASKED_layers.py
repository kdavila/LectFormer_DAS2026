import torch
import torch.nn as nn

from .util import get_raw_patch_neighbors_mask, get_2d_sincos_relative_positional_embed

class TransBlock(nn.Module):
    def __init__(self, latent_dim, n_layers, n_heads, dff_ratio, drop_out, layer_activation, post_layer_norm,
                 mask_neighbors, layerwise_PE, bias):
        super(TransBlock, self).__init__()

        self.__latent_dim = latent_dim
        self.__n_layers = n_layers
        self.__n_heads = n_heads
        self.__dff_ratio = dff_ratio
        self.__drop_out = drop_out
        self.__layer_activation = layer_activation
        self.__bias = bias
        self.__use_layerwise_PE = layerwise_PE

        # self.__pre_mixing = nn.Conv2d(latent_dim, latent_dim, 1)
        # self.__total_token_size = self.__img_seq_len * latent_dim
        # self.__output_size = output_size

        if latent_dim % self.__n_heads != 0:
            raise Exception("The latent dimension cannot be split exactly  by current number of heads")

        dff_size = int(latent_dim * dff_ratio)

        if n_layers == 0:
            # bypass mode
            self.__trans_encoder = nn.Identity()
        else:
            if not self.__use_layerwise_PE:
                # use a standard transformer encoder ...
                enc_layer = nn.TransformerEncoderLayer(latent_dim, n_heads, dff_size, drop_out, layer_activation,
                                                       batch_first=True, norm_first=True, bias=bias)
                self.__trans_encoder = nn.TransformerEncoder(enc_layer, n_layers, enable_nested_tensor=False)
            else:
                layer_list = [
                    nn.TransformerEncoderLayer(latent_dim, n_heads, dff_size, drop_out, layer_activation,
                                               batch_first=True, norm_first=True, bias=bias)
                    for i in range(n_layers)
                ]
                self.__trans_encoder = nn.ModuleList(layer_list)

        if post_layer_norm:
            self.__post_norm = nn.LayerNorm(latent_dim)
        else:
            self.__post_norm = nn.Identity()

        # window is given in N x N format, but this works like padding border
        # convert ....
        # 7x7 window => (7 - 1) // 2 = 3 border
        self._mask_neighbors = (mask_neighbors - 1) // 2

        self._last_nn_mask = None
        self._last_nn_mask_shape = None

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

    @torch._dynamo.disable
    def eager_masked_attention(self, x, pos_embedding, raw_patch_nn_mask):
        if not self.__use_layerwise_PE:
            # PE gets added once at the beginning
            x = x + pos_embedding
            return self.__trans_encoder(x, mask=raw_patch_nn_mask)
        else:
            # PE gets added before each layer ...
            for layer in self.__trans_encoder:
                # assert isinstance(layer, nn.TransformerEncoderLayer)
                x = x + pos_embedding
                x = layer(x, src_mask=raw_patch_nn_mask)

            return x

    @torch._dynamo.disable
    def forward(self, conv_feats):
        # N = batch size, maps_h = num. of height patches, maps_w = num. of width patches
        N, _, maps_h, maps_w = conv_feats.shape

        img_seq_len = maps_h * maps_w
        # pos_embedding = get_sinusoid_encoding(img_seq_len, d_hid=self.__latent_dim)

        # get the 2D attention mask ...
        current_mask_shape = (maps_w, maps_h)
        if current_mask_shape != self._last_nn_mask_shape:
            # different shape, recompute and cache
            self._last_nn_mask = get_raw_patch_neighbors_mask(maps_w, maps_h, self._mask_neighbors)
            self._last_nn_mask_shape = current_mask_shape

        raw_patch_nn_mask = self._last_nn_mask
        raw_patch_nn_mask = raw_patch_nn_mask.to(conv_feats.device)

        # generate the 2D abs positional encoding ...
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

        # pos_embedding = get_2d_sincos_pos_embed(self.__latent_dim, offset_x, offset_y, maps_h, maps_w)
        pos_embedding = get_2d_sincos_relative_positional_embed(self.__latent_dim, offset_y, offset_x, maps_h, maps_w,
                                                                img_h, img_w)

        pos_embedding = pos_embedding.to(conv_feats.device)

        # conv_feats = self.__pre_mixing(conv_feats)
        # now ... split the conv features
        # N x Latent x MapH x MapW -> N x Latent x T
        tempo = conv_feats.reshape(N, self.__latent_dim, img_seq_len)

        # we are using batch_first=True, so the batch should come first, but we need to swap T
        # N x Latent x T  -> N x T x Latent
        tempo = tempo.transpose(1, 2)

        # .... apply encoder ...
        if self.__n_layers > 0:
            enc_output = self.eager_masked_attention(tempo, pos_embedding, raw_patch_nn_mask)
        else:
            # bypass mode
            tempo = tempo + pos_embedding
            enc_output = self.__trans_encoder(tempo)

        # apply post normalization (if any)
        enc_output = self.__post_norm(enc_output)

        # now ... move it back to conv feature map style ...
        # N x T x Latent ->  N x Latent x T
        trans_out = enc_output.transpose(1, 2)

        # assuming that we had maps_h x maps_w,
        # the second param here is the "size" of each slice (size of a row),
        # and the third is the "step" (jump to the next slice)
        trans_out = trans_out.unfold(2, maps_w, maps_w)

        return trans_out

