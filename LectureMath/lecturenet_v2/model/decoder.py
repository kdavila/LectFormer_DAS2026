import random

import torch
from torch import nn as nn

from LM_Tools.configuration.configuration import Configuration
from .res_conv_block import ResConvBlock
from .util import ConvNormalizationHelper, config_get_activation


class FCN_DecoderBlock(nn.Module):
    def __init__(self, kernel_size, activation, in_channels, n_upsample, n_conv_up, n_conv_down, active_skip,
                 skip_concatenate, skip_dropout_rate, use_residual, normalization):
        super(FCN_DecoderBlock, self).__init__()

        assert isinstance(normalization, ConvNormalizationHelper)

        self.__use_residual = use_residual

        first_norm = normalization.get_layer(n_upsample)
        if not self.__use_residual:
            second_norm = normalization.get_layer(n_conv_up)
        else:
            second_norm = None

        self.__transposed_conv = nn.ConvTranspose2d(in_channels, n_upsample, 2, padding=0, stride=2)
        self.__upsample_block = nn.Sequential(
            first_norm,
            activation
        )

        self.__active_skip = active_skip
        self.__skip_concatenate = skip_concatenate
        self.__skip_dropout_rate = skip_dropout_rate

        up_input = n_upsample
        if self.__active_skip:
            if self.__skip_concatenate:
                up_input += n_conv_down
            else:
                if n_upsample != n_conv_down:
                    raise Exception(f"Skip Conn. has {n_conv_down} maps, but up-sampling has {n_upsample}")

        padding = int((kernel_size - 1) / 2)

        if self.__use_residual:
            # use a residual block ...
            self.__conv_up_block = ResConvBlock(up_input, n_conv_up, kernel_size, activation, normalization)
        else:
            # simple convolutional block
            self.__conv_up_block = nn.Sequential(
                nn.Conv2d(up_input, n_conv_up, kernel_size=kernel_size, padding=padding),
                second_norm,
                activation
            )

    def forward(self, x_main, x_skip, x_target):
        x_up = self.__transposed_conv(x_main, output_size=x_target.shape)
        x_up = self.__upsample_block(x_up)
        if self.__active_skip:
            if self.training and self.__skip_dropout_rate is not None and random.random() < self.__skip_dropout_rate:
                # no info passed
                x_up = torch.zeros_like(x_skip, device=x_main.device)
            if self.__skip_concatenate:
                x_up = torch.cat((x_up, x_skip), 1)
            else:
                x_up = x_up + x_skip
        x_up = self.__conv_up_block(x_up)

        return x_up


class FCN_Decoder(nn.Module):
    def __init__(self, kernel_size, activation, in_channels, active_skips, n_upsample_5, n_conv_up_5, n_conv_down_5,
                 n_upsample_4, n_conv_up_4, n_conv_down_4, n_upsample_3, n_conv_up_3, n_conv_down_3,
                 n_upsample_2, n_conv_up_2, n_conv_down_2, n_upsample_1, n_conv_up_1, n_conv_down_1,
                 skip_concatenate, use_residuals, normalization):
        super(FCN_Decoder, self).__init__()

        assert isinstance(normalization, ConvNormalizationHelper)

        padding = int((kernel_size - 1) / 2)

        self.n_conv_up_5 = n_conv_up_5
        self.n_conv_up_4 = n_conv_up_4
        self.n_conv_up_3 = n_conv_up_3
        self.n_conv_up_2 = n_conv_up_2
        # This one is the final output size ...
        self.n_conv_up_1 = n_conv_up_1

        # make sure that all active skips are integers
        self.__active_skips = [int(key) for key in active_skips]
        self.__skip_concatenate = skip_concatenate
        # TODO: this "feature" is locked for now ...
        self.__skip_dropout_rate = None

        self.use_residual = use_residuals

        self.__up_block_5 = FCN_DecoderBlock(kernel_size, activation, in_channels, n_upsample_5, n_conv_up_5,
                                             n_conv_down_5, 5 in self.__active_skips, skip_concatenate,
                                             self.__skip_dropout_rate, self.use_residual, normalization)

        self.__up_block_4 = FCN_DecoderBlock(kernel_size, activation, n_conv_up_5, n_upsample_4, n_conv_up_4,
                                             n_conv_down_4, 4 in self.__active_skips, skip_concatenate,
                                             self.__skip_dropout_rate, self.use_residual, normalization)

        self.__up_block_3 = FCN_DecoderBlock(kernel_size, activation, n_conv_up_4, n_upsample_3, n_conv_up_3,
                                             n_conv_down_3, 3 in self.__active_skips, skip_concatenate,
                                             self.__skip_dropout_rate, self.use_residual, normalization)

        self.__up_block_2 = FCN_DecoderBlock(kernel_size, activation, n_conv_up_3, n_upsample_2, n_conv_up_2,
                                             n_conv_down_2, 2 in self.__active_skips, skip_concatenate,
                                             self.__skip_dropout_rate, self.use_residual, normalization)

        self.__up_block_1 = FCN_DecoderBlock(kernel_size, activation, n_conv_up_2, n_upsample_1, n_conv_up_1,
                                             n_conv_down_1, 1 in self.__active_skips, skip_concatenate,
                                             self.__skip_dropout_rate, self.use_residual, normalization)

        # initializing weights
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # the information that comes from the middle block, plus data from the encoder, and the original input
        x_mid, enc_outs, x0 = x

        enc_l1, enc_l2, enc_l3, enc_l4, enc_l5 = enc_outs
        x_conv1_pre, x_conv1 = enc_l1
        x_conv2_pre, x_conv2 = enc_l2
        x_conv3_pre, x_conv3 = enc_l3
        x_conv4_pre, x_conv4 = enc_l4
        x_conv5_pre, x_conv5 = enc_l5

        # pass the info, use the following order
        # x_main, x_skip, x_target = x
        # like x = (x_mid, x_conv5_pre, x_conv4)
        x_up5 = self.__up_block_5(x_mid, x_conv5_pre, x_conv4)

        # x = (x_up5, x_conv4_pre, x_conv3)
        x_up4 = self.__up_block_4(x_up5, x_conv4_pre, x_conv3)

        # x = (x_up4, x_conv3_pre, x_conv2)
        x_up3 = self.__up_block_3(x_up4, x_conv3_pre, x_conv2)

        # x = (x_up3, x_conv2_pre, x_conv1)
        x_up2 = self.__up_block_2(x_up3, x_conv2_pre, x_conv1)

        # x = (x_up2, x_conv1_pre, x0)
        x_up1 = self.__up_block_1(x_up2, x_conv1_pre, x0)

        return x_up1

    @staticmethod
    def CreateFromConfig(config, use_middle_block):
        assert isinstance(config, Configuration)

        n_conv_down_1 = config.get("LectureNet.Network.Filters.ConvDown_1", 16)
        n_conv_down_2 = config.get("LectureNet.Network.Filters.ConvDown_2", 32)
        n_conv_down_3 = config.get("LectureNet.Network.Filters.ConvDown_3", 64)
        n_conv_down_4 = config.get("LectureNet.Network.Filters.ConvDown_4", 128)
        n_conv_down_5 = config.get("LectureNet.Network.Filters.ConvDown_5", 256)

        if use_middle_block:
            # using a middle conv. block
            dec_in_features = config.get("LectureNet.Network.Filters.Middle", 512)
        else:
            # not using a middle block ...
            dec_in_features = n_conv_down_5

        n_upsample_5 = config.get("LectureNet.Network.Filters.Upsample_5", 256)
        n_conv_up_5 = config.get("LectureNet.Network.Filters.ConvUp_5", 256)

        n_upsample_4 = config.get("LectureNet.Network.Filters.Upsample_4", 128)
        n_conv_up_4 = config.get("LectureNet.Network.Filters.ConvUp_4", 128)

        n_upsample_3 = config.get("LectureNet.Network.Filters.Upsample_3", 64)
        n_conv_up_3 = config.get("LectureNet.Network.Filters.ConvUp_3", 64)

        n_upsample_2 = config.get("LectureNet.Network.Filters.Upsample_2", 32)
        n_conv_up_2 = config.get("LectureNet.Network.Filters.ConvUp_2", 32)

        n_upsample_1 = config.get("LectureNet.Network.Filters.Upsample_1", 16)
        n_conv_up_1 = config.get("LectureNet.Network.Filters.ConvUp_1", 16)

        kernel_size = config.get("LectureNet.Network.Filters.MainKernel", 3)
        activation = config_get_activation(config)

        norm_config = config.get_subconfig("LectureNet.Network.Normalization")
        normalization = ConvNormalizationHelper.FromConfig(norm_config)

        active_skips = config.get_subconfig("LectureNet.Network.Skips.Active").data.keys()
        skips_concatenate = config.get("LectureNet.Network.Skips.Concatenate")

        use_residuals = config.get("LectureNet.Network.Filters.DecoderResiduals", True)

        return FCN_Decoder(
            kernel_size, activation, dec_in_features, active_skips,
            n_upsample_5, n_conv_up_5, n_conv_down_5, n_upsample_4, n_conv_up_4, n_conv_down_4,
            n_upsample_3, n_conv_up_3, n_conv_down_3, n_upsample_2, n_conv_up_2, n_conv_down_2,
            n_upsample_1, n_conv_up_1, n_conv_down_1, skips_concatenate, use_residuals, normalization
        )
