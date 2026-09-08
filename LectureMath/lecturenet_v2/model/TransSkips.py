from torch import nn as nn

from .FLEX_layers import FlexTransBlock
from .MASKED_layers import TransBlock
from .NAT_layers import NATTransBlock
from .Shifted_layers import ShiftedTransBlock
from .WinFaster_layers import WINDTransBlock
from .util import config_get_activation


class FCN_TransSkips(nn.Module):
    def __init__(self, n_conv_maps, patch_size, active_config, bypass_mode, skips_concatenate, tf_activation):
        super(FCN_TransSkips, self).__init__()

        self.active_skips = active_config.get_names()

        self.skips_concatenate = skips_concatenate
        self.bypass_mode = bypass_mode

        # final_post_layer_norm = False
        # n_layers = 6  # 6; So far 8 is best! but larger could help if more data is available!
        # diff_ratio = 4.0  # 4 is a good default, and lower seems to hurt
        # n_heads  = 16  # 8 has been used, but seems not great for width=48.
        # dropout = 0.1

        # sub_patches
        level_patch_size = patch_size
        tempo_skip_downs = []
        tempo_down_unshuffles = []
        tempo_skip_ups = []
        tempo_up_shuffles = []
        tempo_transformers = []
        for idx in range(5):
            level_id = str(idx + 1)
            if (level_id in self.active_skips) and not self.bypass_mode:
                # an active skip connection ...
                level_subconfig = active_config.get_subconfig(f"{level_id}")

                level_sub_patches = level_subconfig.get("SubPatches.PerSide")

                # true = make all a single patch via concatenation,
                # false = create many independent mini-patches
                level_sub_patch_concatenate = level_subconfig.get("SubPatches.Concatenate")

                tf_type = level_subconfig.get("Transformers.Type").lower()
                tf_use_NAT = tf_type == "nat"
                tf_use_WIND = tf_type == "wind"
                tf_use_WIND_FAST = tf_type == "windfast"
                tf_use_FLEX = tf_type == "flex"
                tf_use_SHIFTED = tf_type == "shifted"
                tf_n_layers = level_subconfig.get("Transformers.Layers")
                tf_diff_ratio = level_subconfig.get("Transformers.ExpRatio")
                tf_n_heads = level_subconfig.get("Transformers.Heads")
                tf_dropout = level_subconfig.get("Transformers.Dropout")
                tf_final_post_norm = level_subconfig.get("Transformers.FinalPostNorm")
                tf_mask_neighbors = level_subconfig.get("Transformers.MaskNeighbors")

                if level_subconfig.contains("Transformers.RelPosBias"):
                    tf_use_rpb = level_subconfig.get("Transformers.RelPosBias")
                else:
                    tf_use_rpb = False
                if level_subconfig.contains("Transformers.LayerAbsEnc"):
                    tf_use_layerwise_abs_pos_enc = level_subconfig.get("Transformers.LayerAbsEnc")
                else:
                    tf_use_layerwise_abs_pos_enc = False
                if level_subconfig.contains("Transformers.ShiftedHybrid"):
                    tf_use_shifted_hybrid_layers = level_subconfig.get("Transformers.ShiftedHybrid")
                else:
                    tf_use_shifted_hybrid_layers = False

                # print(f"{level_id} is active!")

                # 1: 32 (large 1x1), 16 (large, 2x2), 8 (large, 4x4), 4 (small, 4x4)
                # 2: 16 (large 1x1), 8 (large, 2x2), 4 (small, 2x2)
                # 3:  8 (Large, 1x1), 4 (Large, 2x2), 2 (small, 2x2)
                # 4:  4 (Large, 1x1), 2 (small, 1x1)
                # 5:  2 (Large, 1x1), 1 (small, 1x1)
                sub_patch_size = level_patch_size // level_sub_patches

                if sub_patch_size > 1:
                    tempo_skip_downs.append(nn.AvgPool2d(sub_patch_size))
                    # TODO: these could be Transposed convolutions ...
                    tempo_skip_ups.append(nn.Upsample(scale_factor=(sub_patch_size, sub_patch_size), mode='nearest'))
                else:
                    tempo_skip_downs.append(nn.Identity())
                    tempo_skip_ups.append(nn.Identity())

                if level_sub_patches > 1 and level_sub_patch_concatenate:
                    # this will make the sub-patch features to be concatenated (256x4x4 -> 1024x1x1)
                    tempo_down_unshuffles.append(nn.PixelUnshuffle(downscale_factor=level_sub_patches))
                    tempo_up_shuffles.append(nn.PixelShuffle(upscale_factor=level_sub_patches))
                    tf_dim = n_conv_maps[level_id] * level_sub_patches * level_sub_patches
                    final_neighborhood = tf_mask_neighbors
                    final_heads = tf_n_heads
                else:
                    # not needed more sub-patches at this level OR using mini-patches
                    # in the second scenario, the patch size will be smaller, with fewer features
                    tempo_down_unshuffles.append(nn.Identity())
                    tempo_up_shuffles.append(nn.Identity())
                    tf_dim = n_conv_maps[level_id]

                    # NOTE, earlier versions expanded tf_mask_neighbors automatically as this gave better results
                    #      but now this expansion has to be manually handled in the configuration!!!
                    # final_neighborhood = tf_mask_neighbors * level_sub_patches - 1
                    final_neighborhood = tf_mask_neighbors

                    final_heads = tf_n_heads // (level_sub_patches * level_sub_patches)

                # enabled by default. Not exposed in the main configuration, but easy to do if needed.
                qkv_bias = True
                proj_bias = True

                print(f"Skip Connection at Level={level_id}")
                print(f" - Layers ={tf_n_layers}")
                print(f" - Neighborhood={final_neighborhood}")
                print(f" - Concat ={level_sub_patch_concatenate}")
                if tf_use_SHIFTED:
                    print(f" - Using Shifted Windows blocks (RPB={tf_use_rpb})")
                    level_tf = ShiftedTransBlock(tf_dim, tf_n_layers, final_heads, tf_diff_ratio, tf_dropout,
                                                 tf_activation, tf_final_post_norm, final_neighborhood, qkv_bias,
                                                 proj_bias, tf_use_rpb, tf_use_shifted_hybrid_layers)
                elif tf_use_FLEX:
                    print(f" - Using Flex-based blocks (RPB={tf_use_rpb}")
                    level_tf = FlexTransBlock(tf_dim, tf_n_layers, final_heads, tf_diff_ratio, tf_dropout,
                                              tf_activation, tf_final_post_norm, final_neighborhood, qkv_bias,
                                              proj_bias, tf_use_rpb)
                elif tf_use_WIND_FAST:
                    print(f" - Using Faster Window-based blocks (RPB={tf_use_rpb})")
                    level_tf = WINDTransBlock(True, tf_dim, tf_n_layers, final_heads, tf_diff_ratio, tf_dropout,
                                              tf_activation, tf_final_post_norm, final_neighborhood, qkv_bias,
                                              proj_bias, tf_use_rpb)
                elif tf_use_WIND:
                    print(f" - Using Window-based blocks (RPB={tf_use_rpb})")
                    level_tf = WINDTransBlock(False, tf_dim, tf_n_layers, final_heads, tf_diff_ratio, tf_dropout,
                                              tf_activation, tf_final_post_norm, final_neighborhood, qkv_bias,
                                              proj_bias, tf_use_rpb)
                elif tf_use_NAT:
                    print(f" - Using NAT blocks (RPB={tf_use_rpb})")
                    # NAT-based transformer, with special NAT kernels for local attention
                    level_tf = NATTransBlock(tf_dim, tf_n_layers, final_heads, tf_diff_ratio, qkv_bias, tf_dropout,
                                             tf_activation, tf_final_post_norm, final_neighborhood, tf_use_rpb)
                else:
                    print(f" - Using Masked blocks")
                    print(f" - Layer-wise PE: {tf_use_layerwise_abs_pos_enc}")
                    # regular transformer self-attention using Local attention via Masked Neighborhoods
                    level_tf = TransBlock(tf_dim, tf_n_layers, final_heads, tf_diff_ratio, tf_dropout, tf_activation,
                                          tf_final_post_norm, final_neighborhood, tf_use_layerwise_abs_pos_enc,
                                          qkv_bias)

                tempo_transformers.append(level_tf)
            else:
                # skip not used, create dummy modules ...
                tempo_skip_downs.append(nn.Identity())
                tempo_skip_ups.append(nn.Identity())
                tempo_down_unshuffles.append(nn.Identity())
                tempo_up_shuffles.append(nn.Identity())
                tempo_transformers.append(nn.Identity())

            level_patch_size = level_patch_size // 2

        self.down_subsampling = nn.ModuleList(tempo_skip_downs)
        self.down_unshuffling = nn.ModuleList(tempo_down_unshuffles)

        self.up_upsampling = nn.ModuleList(tempo_skip_ups)
        self.up_shuffling = nn.ModuleList(tempo_up_shuffles)

        self.mid_blocks = nn.ModuleList(tempo_transformers)

    def forward(self, enc_outs):
        tempo_dec_levels = []
        for idx in range(5):
            # get features for current level ...
            # (pre-max pooling, post-max pooling) -> using pre-max for smaller patches!
            x_conv_pre, x_conv = enc_outs[idx]

            if str(idx + 1) in self.active_skips:
                # this is an active skip
                # check if bypass the transformer or not ...
                if self.bypass_mode:
                    # simply pass the raw data
                    dec_level = x_conv_pre, x_conv
                else:
                    # use the transformer
                    # Downscale features for this level ...
                    skip_feats_in = self.down_subsampling[idx](x_conv_pre)
                    # and then un-shuffle as required to concatenate the sub-patch features
                    tran_in_fts = self.down_unshuffling[idx](skip_feats_in)
                    # run the transformer layer
                    tran_out_fts = self.mid_blocks[idx](tran_in_fts)
                    # re-organize features within patch
                    skip_feats_out = self.up_shuffling[idx](tran_out_fts)
                    # upscale the features for later concatenation ..
                    # Warning: unlike the original lecturenet, this will fail if the size is not perfectly divisible by 32
                    skip_feats_out = self.up_upsampling[idx](skip_feats_out)

                    # this will be passed to the decoder ...
                    dec_level = skip_feats_out, x_conv
            else:
                # not active, bypass ...
                dec_level = None, x_conv

            tempo_dec_levels.append(dec_level)

        return tuple(tempo_dec_levels)

    @staticmethod
    def CreateFromConfig(config):
        conv_maps = {
            "1": config.get("LectureNet.Network.Filters.ConvDown_1", 16),
            "2": config.get("LectureNet.Network.Filters.ConvDown_2", 32),
            "3": config.get("LectureNet.Network.Filters.ConvDown_3", 64),
            "4": config.get("LectureNet.Network.Filters.ConvDown_4", 128),
            "5": config.get("LectureNet.Network.Filters.ConvDown_5", 256)
        }

        skip_config = config.get_subconfig("LectureNet.Network.Skips")
        patch_size = skip_config.get("PatchSize")
        skips_concatenate = skip_config.get("Concatenate")
        bypass_mode = skip_config.get("BypassMode")
        active_config = skip_config.get_subconfig("Active")

        activation = config_get_activation(config)

        fcn_skips = FCN_TransSkips(conv_maps, patch_size, active_config, bypass_mode, skips_concatenate, activation)

        return fcn_skips
