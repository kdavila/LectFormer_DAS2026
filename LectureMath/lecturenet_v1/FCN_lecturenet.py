
import math
import cv2
import numpy as np
import platform

import PIL
import PIL.Image
from PIL import ImageOps

import torch
import torch.nn as nn

from torchvision import transforms
import torchvision.transforms.functional as TF

from LM_Tools.configuration.configuration import Configuration

# to be used as decorator for platform-dependent compiler exceptions
compile_disable_if_windows = (
    torch.compiler.disable
    if platform.system() == "Windows"
    else lambda f: f
)


class FCN_ConvBlock_V1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, use_pooling):
        super(FCN_ConvBlock_V1, self).__init__()

        padding = (kernel_size - 1) // 2

        self.use_pooling = use_pooling

        self.conv_down_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation
        )
        if self.use_pooling:
            # add a max-pooling layer
            self.conv_block_pool = nn.MaxPool2d(2, return_indices=False)
        else:
            self.conv_block_pool = nn.Identity()

        #  initialize conv layer weights ...
        # nn.init.xavier_normal_(self.conv_down_block[0].weight)
        # nn.init.constant_(self.conv_down_block[0].bias, 0.0)
        nn.init.kaiming_normal_(self.conv_down_block[0].weight)
        nn.init.zeros_(self.conv_down_block[0].bias)
        #  initialize batch norm layer weights ...
        self.conv_down_block[1].weight.data.normal_(1.0, 0.02)
        self.conv_down_block[1].bias.data.fill_(0)

    @compile_disable_if_windows
    def maybe_eager_pooling(self, x_conv_pre):
        return self.conv_block_pool(x_conv_pre)

    def forward(self, x):
        x_conv_pre = self.conv_down_block(x)
        if self.use_pooling:
            # run in a region where compilation is disabled for windows
            x_conv = self.maybe_eager_pooling(x_conv_pre)
        else:
            # just use the layer as given
            x_conv = self.conv_block_pool(x_conv_pre)

        return x_conv_pre, x_conv


class FCN_ConvBlock_V2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, use_pooling):
        super(FCN_ConvBlock_V2, self).__init__()

        self.sub_block_1 = FCN_ConvBlock_V1(in_channels, out_channels, kernel_size, activation, False)
        self.sub_block_2 = FCN_ConvBlock_V1(out_channels, out_channels, kernel_size, activation, use_pooling)

    def forward(self, x):
        _, x1_conv = self.sub_block_1(x)
        x2_pre, x2_conv = self.sub_block_2(x1_conv)

        return x2_pre, x2_conv


class FCN_Encoder(nn.Module):
    def __init__(self, channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5,
                 kernel_size, activation, use_double_blocks):
        super(FCN_Encoder, self).__init__()

        if use_double_blocks:
            # deeper network, with two activation layers per scale ...
            self.conv_down_block_1 = FCN_ConvBlock_V2(channels, n_conv_down_1, kernel_size, activation, True)
            self.conv_down_block_2 = FCN_ConvBlock_V2(n_conv_down_1, n_conv_down_2, kernel_size, activation, True)
            self.conv_down_block_3 = FCN_ConvBlock_V2(n_conv_down_2, n_conv_down_3, kernel_size, activation, True)
            self.conv_down_block_4 = FCN_ConvBlock_V2(n_conv_down_3, n_conv_down_4, kernel_size, activation, True)
            self.conv_down_block_5 = FCN_ConvBlock_V2(n_conv_down_4, n_conv_down_5, kernel_size, activation, True)
        else:
            # Original FCN-Lecture Design based on simpler blocks, with only one activation layer per scale ...
            self.conv_down_block_1 = FCN_ConvBlock_V1(channels, n_conv_down_1, kernel_size, activation, True)
            self.conv_down_block_2 = FCN_ConvBlock_V1(n_conv_down_1, n_conv_down_2, kernel_size, activation, True)
            self.conv_down_block_3 = FCN_ConvBlock_V1(n_conv_down_2, n_conv_down_3, kernel_size, activation, True)
            self.conv_down_block_4 = FCN_ConvBlock_V1(n_conv_down_3, n_conv_down_4, kernel_size, activation, True)
            self.conv_down_block_5 = FCN_ConvBlock_V1(n_conv_down_4, n_conv_down_5, kernel_size, activation, True)

    def forward(self, x0):
        x_conv1_pre, x_conv1 = self.conv_down_block_1(x0)
        x_conv2_pre, x_conv2 = self.conv_down_block_2(x_conv1)
        x_conv3_pre, x_conv3 = self.conv_down_block_3(x_conv2)
        x_conv4_pre, x_conv4 = self.conv_down_block_4(x_conv3)
        x_conv5_pre, x_conv5 = self.conv_down_block_5(x_conv4)

        return [
            (x_conv1_pre, x_conv1),
            (x_conv2_pre, x_conv2),
            (x_conv3_pre, x_conv3),
            (x_conv4_pre, x_conv4),
            (x_conv5_pre, x_conv5)
        ]


class FCN_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_upsample, n_conv_down, n_conv_up, kernel_size, activation, concat_skips):
        super(FCN_DecoderBlock, self).__init__()

        padding = (kernel_size - 1) // 2

        self.__concat_skips = concat_skips

        self.transposed_conv = nn.ConvTranspose2d(in_channels, n_upsample, 2, padding=0, stride=2)
        self.upsample_block = nn.Sequential(
            nn.BatchNorm2d(n_upsample),
            activation
        )

        if self.__concat_skips:
            # original FCN-LectureNet behavior, more parameters
            conv_up_maps = n_upsample + n_conv_down
        else:
            # FCN-LectureNet v1.1. behavior, fewer parameters, treated as a residual connection
            conv_up_maps = n_upsample

        self.conv_up_block = nn.Sequential(
            nn.Conv2d(conv_up_maps, n_conv_up, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_conv_up),
            activation
        )

        # initialize transpose conv. weights ...
        # nn.init.xavier_normal_(self.transposed_conv.weight)
        # nn.init.constant_(self.transposed_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.transposed_conv.weight)
        nn.init.zeros_(self.transposed_conv.bias)
        # initialize conv. weights ...
        # nn.init.xavier_normal_(self.conv_up_block[0].weight)
        # nn.init.constant_(self.transposed_conv[0].bias, 0.0)
        nn.init.kaiming_normal_(self.conv_up_block[0].weight)
        nn.init.zeros_(self.conv_up_block[0].bias)
        # initialize batch norm layer weights ...
        self.upsample_block[0].weight.data.normal_(1.0, 0.02)
        self.upsample_block[0].bias.data.fill_(0)
        self.conv_up_block[1].weight.data.normal_(1.0, 0.02)
        self.conv_up_block[1].bias.data.fill_(0)

    def forward(self, x_conv_in, x_size, x_conv_pre):
        # initial up-sampling ...
        x_up = self.transposed_conv(x_conv_in, output_size=x_size.shape)
        # followed by normalization and activation ...
        x_up = self.upsample_block(x_up)

        if self.__concat_skips:
            # more features available
            x_up = torch.cat((x_up, x_conv_pre), 1)
        else:
            # treat skip as residual, force feature alignment!
            x_up = x_up + x_conv_pre

        x_up = self.conv_up_block(x_up)

        return x_up


class FCN_Decoder(nn.Module):
    def __init__(self, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5, mid_block,
                 n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4, n_upsample_3, n_conv_up_3,
                 n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1, kernel_size, activation, use_double_blocks,
                 use_skip_concat):
        super(FCN_Decoder, self).__init__()

        if use_double_blocks:
            # middle conv block
            self.mid_block = FCN_ConvBlock_V2(n_conv_down_5, mid_block, kernel_size, activation, False)
        else:
            # middle conv block (original architecture)
            self.mid_block = FCN_ConvBlock_V1(n_conv_down_5, mid_block, kernel_size, activation, False)

        self.up_block_5 = FCN_DecoderBlock(mid_block, n_upsample_5, n_conv_down_5, n_conv_up_5, kernel_size,
                                           activation, use_skip_concat)

        self.up_block_4 = FCN_DecoderBlock(n_conv_up_5, n_upsample_4, n_conv_down_4, n_conv_up_4, kernel_size,
                                           activation, use_skip_concat)

        self.up_block_3 = FCN_DecoderBlock(n_conv_up_4, n_upsample_3, n_conv_down_3, n_conv_up_3, kernel_size,
                                           activation, use_skip_concat)

        self.up_block_2 = FCN_DecoderBlock(n_conv_up_3, n_upsample_2, n_conv_down_2, n_conv_up_2, kernel_size,
                                           activation, use_skip_concat)

        self.up_block_1 = FCN_DecoderBlock(n_conv_up_2, n_upsample_1, n_conv_down_1, n_conv_up_1, kernel_size,
                                           activation, use_skip_concat)

    def forward(self, x0, encoder_outputs):
        x_conv1_pre, x_conv1 = encoder_outputs[0]
        x_conv2_pre, x_conv2 = encoder_outputs[1]
        x_conv3_pre, x_conv3 = encoder_outputs[2]
        x_conv4_pre, x_conv4 = encoder_outputs[3]
        x_conv5_pre, x_conv5 = encoder_outputs[4]

        # run mid-block
        _, x_mid = self.mid_block(x_conv5)

        # then the up-sampling blocks with skip connections ...
        x_up5 = self.up_block_5(x_mid, x_conv4, x_conv5_pre)
        x_up4 = self.up_block_4(x_up5, x_conv3, x_conv4_pre)
        x_up3 = self.up_block_3(x_up4, x_conv2, x_conv3_pre)
        x_up2 = self.up_block_2(x_up3, x_conv1, x_conv2_pre)
        x_up1 = self.up_block_1(x_up2, x0, x_conv1_pre)

        return x_up1


class FCN_AutoEncoder(nn.Module):
    def __init__(self, channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5,
                 mid_block, n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4, n_upsample_3, n_conv_up_3,
                 n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1, kernel_size, rec_kernel_size, activation,
                 use_double_blocks, skip_concatenate, prc_masked_features):
        super(FCN_AutoEncoder, self).__init__()

        # initial convolutions ...
        padding = int((kernel_size - 1) / 2)

        self.__prc_masked_features = prc_masked_features

        if self.__prc_masked_features > 0:
            self.__up_2x = nn.Upsample(scale_factor=(2, 2), mode='nearest')
        else:
            self.__up_2x = None

        self.encoder = FCN_Encoder(channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5,
                                   kernel_size, activation, use_double_blocks)

        """
        if self.__use_middle_block:
            # using a middle conv. block
            self.mid_block = nn.Sequential(
                nn.Conv2d(n_conv_down_5, mid_block, stride=1, kernel_size=kernel_size, padding=padding),
                mid_first_norm,
                activation,
                nn.Conv2d(mid_block, mid_block, stride=1, kernel_size=kernel_size, padding=padding),
                mid_second_norm,
                activation,
            )
            nn.init.xavier_normal_(self.mid_block[0].weight)
            nn.init.xavier_normal_(self.mid_block[3].weight)
            nn.init.constant_(self.mid_block[0].bias, 0.0)
            nn.init.constant_(self.mid_block[3].bias, 0.0)

            dec_in_features = mid_block
        else:
            # not using a middle block ...
            self.mid_block = None
            dec_in_features = n_conv_down_5
        """

        self.decoder = FCN_Decoder(n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5, mid_block,
                                   n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4, n_upsample_3, n_conv_up_3,
                                   n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1, kernel_size, activation,
                                   use_double_blocks, skip_concatenate)

        rec_padding = int((rec_kernel_size - 1) / 2)

        # nn.Conv2d(n_conv_up_1, 32, stride=1, kernel_size=rec_kernel_size, padding=rec_padding),
        # normalization.get_layer(32),
        # activation,
        # nn.Conv2d(32, 3, stride=1, kernel_size=rec_kernel_size, padding=rec_padding),
        self.conv_reconstruct = nn.Sequential(
            nn.Conv2d(n_conv_up_1, 3, stride=1, kernel_size=rec_kernel_size, padding=rec_padding),
            nn.Tanh(),
        )
        nn.init.xavier_normal_(self.conv_reconstruct[0].weight)
        nn.init.constant_(self.conv_reconstruct[0].bias, 0.0)

        self.last_mask = None

    def get_prc_masked_features(self):
        return self.__prc_masked_features

    def forward(self, x0):
        # encode ...
        enc_outs = self.encoder(x0)

        if self.__prc_masked_features > 0:
            # mask the features ...
            _, x_conv5 = enc_outs[4]
            _, _, ph, pw = x_conv5.shape
            mask = (torch.rand((1, 1, ph, pw)) >= self.__prc_masked_features).type(x_conv5.dtype).to(x0.device)
            mask_2x = self.__up_2x(mask)
            mask_4x = self.__up_2x(mask_2x)
            mask_8x = self.__up_2x(mask_4x)
            mask_16x = self.__up_2x(mask_8x)
            mask_32x = self.__up_2x(mask_16x)

            self.last_mask = mask_32x[0, 0].detach().clone().cpu().numpy()

            # print((enc_outs[0][0].shape, mask_32x[0, 0].shape))
            enc_feats = [
                [enc_outs[0][0] * mask_32x[0, 0], enc_outs[0][1] * mask_16x[0, 0]],
                [enc_outs[1][0] * mask_16x[0, 0], enc_outs[1][1] * mask_8x[0, 0]],
                [enc_outs[2][0] * mask_8x[0, 0], enc_outs[2][1] * mask_4x[0, 0]],
                [enc_outs[3][0] * mask_4x[0, 0], enc_outs[3][1] * mask_2x[0, 0]],
                [enc_outs[4][0] * mask_2x[0, 0], enc_outs[4][1] * mask[0, 0]]
            ]
        else:
            # use the original, raw features (no masking)
            enc_feats = enc_outs

        # decoder ...
        x_up1 = self.decoder(x0, enc_feats)

        # final reconstruction branch
        raw_img = self.conv_reconstruct(x_up1)

        return raw_img

    def reconstruct(self, PIL_image):
        # make sure that each size is a multiple of ...32!
        divisor = 2 ** 5  #
        pre_pad_w, pre_pad_h = PIL_image.size
        padded_w = math.ceil(pre_pad_w / divisor) * divisor
        padded_h = math.ceil(pre_pad_h / divisor) * divisor

        pad_w = padded_w - pre_pad_w
        pad_h = padded_h - pre_pad_h
        padding = (0, 0, pad_w, pad_h)
        PIL_image = ImageOps.expand(PIL_image, border=padding, fill=0)

        # assert isinstance(PIL_image, Image)
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            res = self.forward(batch_img)

        if using_cuda:
            res = res.cpu()

        # De-normalize ...
        image = res[0].numpy()

        # for tanh ...
        image = self.from_img_space_to_cv2(image)

        return image

    def from_img_space_to_cv2(self, image):
        # for tanh
        # tranpose .. from (Channels, H, W) to (H, W, Channels)
        image = np.transpose(image, (1, 2, 0))
        # scale from [-1, 1] to [-0.5, 0.5]
        image *= 0.5
        # translate from [-0.5, 0.5] to [0.0, 1.0]
        image += 0.5

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        # these should not be needed for Tanh
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    @staticmethod
    def CreateFromConfig(config, in_channels, original_arch,  prc_feat_masking):

        use_double_blocks = not original_arch
        concat_skips = original_arch

        n_convs_down_1 = config.get("LectureNet.Network.Filters.ConvDown_1", 16)
        n_convs_down_2 = config.get("LectureNet.Network.Filters.ConvDown_2", 32)
        n_convs_down_3 = config.get("LectureNet.Network.Filters.ConvDown_3", 64)
        n_convs_down_4 = config.get("LectureNet.Network.Filters.ConvDown_4", 128)
        n_convs_down_5 = config.get("LectureNet.Network.Filters.ConvDown_5", 256)

        n_convs_mid = config.get("LectureNet.Network.Filters.Middle", 512)

        n_upscale_5 = config.get("LectureNet.Network.Filters.Upsample_5", 256)
        n_convs_up_5 = config.get("LectureNet.Network.Filters.ConvUp_5", 256)

        n_upscale_4 = config.get("LectureNet.Network.Filters.Upsample_4", 128)
        n_convs_up_4 = config.get("LectureNet.Network.Filters.ConvUp_4", 128)

        n_upscale_3 = config.get("LectureNet.Network.Filters.Upsample_3", 64)
        n_convs_up_3 = config.get("LectureNet.Network.Filters.ConvUp_3", 64)

        n_upscale_2 = config.get("LectureNet.Network.Filters.Upsample_2", 32)
        n_convs_up_2 = config.get("LectureNet.Network.Filters.ConvUp_2", 32)

        n_upscale_1 = config.get("LectureNet.Network.Filters.Upsample_1", 16)
        n_convs_up_1 = config.get("LectureNet.Network.Filters.ConvUp_1", 16)

        pix_kernel_size = config.get("LectureNet.Network.Filters.PixelKernel", 3)
        kernel_size = config.get("LectureNet.Network.Filters.MainKernel", 3)

        activation = FCN_LectureNet.GetActivation(config)

        auto_encoder = FCN_AutoEncoder(in_channels, n_convs_down_1, n_convs_down_2, n_convs_down_3, n_convs_down_4,
                                       n_convs_down_5, n_convs_mid, n_upscale_5, n_convs_up_5,
                                       n_upscale_4, n_convs_up_4, n_upscale_3, n_convs_up_3,
                                       n_upscale_2, n_convs_up_2, n_upscale_1, n_convs_up_1, kernel_size,
                                       pix_kernel_size, activation, use_double_blocks, concat_skips, prc_feat_masking)

        return auto_encoder


class FCN_BinarizerBranch(nn.Module):
    def __init__(self, channels, kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2, activation):
        super(FCN_BinarizerBranch, self).__init__()

        padding = int((kernel_size - 1) / 2)

        # .... now use 1D convolutions ...
        inputs_conv_pixels_1 = channels + n_conv_up_1
        # inputs_conv_pixels_1 = channels + n_conv_up_1 + 1
        self.conv_pixels_1 = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_1, n_pmaps_1, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_pmaps_1),
            activation
        )
        nn.init.xavier_normal_(self.conv_pixels_1[0].weight)
        nn.init.constant_(self.conv_pixels_1[0].bias, 0.0)
        self.conv_pixels_1[1].weight.data.normal_(1.0, 0.02)
        self.conv_pixels_1[1].bias.data.fill_(0)

        inputs_conv_pixels_2 = channels + n_pmaps_1
        self.conv_pixels_2 = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_2, n_pmaps_2, stride=1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_pmaps_2),
            activation
        )
        nn.init.xavier_normal_(self.conv_pixels_2[0].weight)
        nn.init.constant_(self.conv_pixels_2[0].bias, 0.0)
        self.conv_pixels_2[1].weight.data.normal_(1.0, 0.02)
        self.conv_pixels_2[1].bias.data.fill_(0)

        # output 1: binary
        inputs_conv_pixels_3 = channels + n_pmaps_2
        self.conv_out = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_3, 1, stride=1, kernel_size=kernel_size, padding=padding),
        )
        # TODO: update if more layers are added!!
        nn.init.xavier_normal_(self.conv_out[0].weight)
        nn.init.constant_(self.conv_out[0].bias, 0.0)

    def forward(self, diff_img, x_up1):
        # First Branch: Binary Image
        # ... add the input maps ...
        x_pixels_0 = torch.cat((diff_img, x_up1), 1)

        # ... apply convolutional block #1
        x_pixels_1 = self.conv_pixels_1(x_pixels_0)

        # ... add the input maps again ...
        x_pixels_1 = torch.cat((diff_img, x_pixels_1), 1)

        # ... apply convolutional block #2
        x_pixels_2 = self.conv_pixels_2(x_pixels_1)

        # ... add the input maps ...
        x_pixels_2 = torch.cat((diff_img, x_pixels_2), 1)

        # ... get last combination (NO SIGMOID)
        output = self.conv_out(x_pixels_2)

        return output


class FCN_LectureNet(nn.Module):
    def __init__(self, channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5, mid_block,
                 n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4, n_upsample_3, n_conv_up_3,
                 n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1, kernel_size,
                 n_pmaps_1, n_pmaps_2, pixel_kernel_size, activation, reconstruction_mode, original_arch):
        super(FCN_LectureNet, self).__init__()

        # this flag is currently used to keep the old behavior (ACCESS 2021 version)
        # set to TRUE to run FCN-LectureNet (V 1.0)
        self.original_version = original_arch
        if self.original_version:
            print("-  FCN-LectureNet version 1.0 in use")
        else:
            print("-  FCN-LectureNet version 1.1 in use")

        # initial convolutions ...
        padding = int((kernel_size - 1) / 2)

        if self.original_version:
            # original, shallower encoder ...
            self.encoder = FCN_Encoder(channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4,
                                       n_conv_down_5, kernel_size, activation, False)

            # middle block is shallow (1 activation)
            # decoder using concatenation ...
            self.decoder = FCN_Decoder(n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5,
                                       mid_block, n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4,
                                       n_upsample_3, n_conv_up_3, n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1,
                                       kernel_size, activation, False, True)
        else:
            # version 1.1. - deeper encoder ...
            self.encoder = FCN_Encoder(channels, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4,
                                       n_conv_down_5, kernel_size, activation, True)

            # middle block is deeper (2 activations)
            # decoder using addition (residual connections) ...
            self.decoder = FCN_Decoder(n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4, n_conv_down_5,
                                       mid_block, n_upsample_5, n_conv_up_5, n_upsample_4, n_conv_up_4,
                                       n_upsample_3, n_conv_up_3, n_upsample_2, n_conv_up_2, n_upsample_1, n_conv_up_1,
                                       kernel_size, activation, True, False)

        # FIRST OUTPUT BRANCH: BINARIZATION
        self.conv_binarizer = None

        # SECOND OUTPUT BRANCH: Text Masks
        self.conv_text_mask_out = None

        self.set_main_branches(channels, pixel_kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2, activation)

        # THIRD OUTPUT BRANCH: reconstruction
        if self.original_version:
            self.conv_reconstruct = nn.Sequential(
                nn.Conv2d(n_conv_up_1, 3, stride=1, kernel_size=kernel_size, padding=padding),
                nn.Tanh(),
            )
            nn.init.xavier_normal_(self.conv_reconstruct[0].weight)
            nn.init.constant_(self.conv_reconstruct[0].bias, 0.0)
        else:
            # updated version
            """
            # new version - R
            self.conv_reconstruct = nn.Sequential(
                ASPP(n_conv_up_1, 9, [1, 2], n_conv_up_1 // 2, nn.GELU(), True),
                ASPP(n_conv_up_1, 9, [1, 2], n_conv_up_1 // 2, nn.GELU(), True),
                ASPP(n_conv_up_1, 9, [1, 2], n_conv_up_1 // 4, nn.GELU(), True),
                ASPP(n_conv_up_1 // 2, 9, [1, 2], n_conv_up_1 // 4, nn.GELU(), True),
                nn.Conv2d(n_conv_up_1 // 2, 3, stride=1, kernel_size=9, padding=4, dilation=1),
                nn.Tanh()
            )
            nn.init.xavier_normal_(self.conv_reconstruct[4].weight)
            """
            # TEMPO, this is only for testing
            self.conv_reconstruct = nn.Sequential(
                nn.Conv2d(n_conv_up_1, 3, stride=1, kernel_size=kernel_size, padding=padding),
                nn.Tanh(),
            )
            nn.init.xavier_normal_(self.conv_reconstruct[0].weight)
            print("Warning: Using temporary shallower reconstruction branch for Version 1.1")

        self.reconstruction_mode = reconstruction_mode

    def set_main_branches(self, channels, kernel_size,  n_conv_up_1, n_pmaps_1, n_pmaps_2, activation):
        padding = int((kernel_size - 1) / 2)

        # FIRST OUTPUT BRANCH: BINARIZATION
        self.conv_binarizer = FCN_BinarizerBranch(channels, kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2, activation)


        # SECOND OUTPUT BRANCH: Text Masks
        if self.original_version:
            self.conv_text_mask_out = nn.Sequential(
                nn.Conv2d(n_conv_up_1, 1, stride=1, kernel_size=kernel_size, padding=padding),
            )
            nn.init.xavier_normal_(self.conv_text_mask_out[0].weight)
            nn.init.constant_(self.conv_text_mask_out[0].bias, 0.0)
        else:

            self.conv_text_mask_out = nn.Sequential(
                nn.Conv2d(n_conv_up_1, 1, stride=1, kernel_size=5, padding=2),
            )
            nn.init.xavier_normal_(self.conv_text_mask_out[0].weight)
            nn.init.constant_(self.conv_text_mask_out[0].bias, 0.0)
            print("Warning: Using temporary shallower text detection branch for Version 1.1")

            """
            self.conv_text_mask_out = nn.Sequential(
                ASPP(n_conv_up_1, 9, [1, 2, 3, 4], n_conv_up_1 // 4, nn.GELU(), True),
                ASPP(n_conv_up_1, 9, [1, 2, 3, 4], n_conv_up_1 // 4, nn.GELU(), True),
                ASPP(n_conv_up_1, 9, [1, 2, 3, 4], n_conv_up_1 // 8, nn.GELU(), True),
                ASPP(n_conv_up_1 // 2, 9, [1, 2, 3, 4], n_conv_up_1 // 8, nn.GELU(), True),
                nn.Conv2d(n_conv_up_1 // 2, 1, stride=1, kernel_size=9, padding=4, dilation=1)
            )
            nn.init.xavier_normal_(self.conv_text_mask_out[4].weight)
            """
            """
            # version R
            self.conv_text_mask_out = nn.Sequential(
                nn.Conv2d(n_conv_up_1, n_conv_up_1, stride=1, kernel_size=9, padding=4, dilation=1),  #
                nn.BatchNorm2d(n_conv_up_1),
                activation,

                nn.Conv2d(n_conv_up_1, n_conv_up_1, stride=1, kernel_size=9, padding=4, dilation=1),  #
                nn.BatchNorm2d(n_conv_up_1),
                activation,

                nn.Conv2d(n_conv_up_1, n_conv_up_1 // 2, stride=1, kernel_size=9, padding=4, dilation=1),
                nn.BatchNorm2d(n_conv_up_1 // 2),
                activation,

                nn.Conv2d(n_conv_up_1 // 2, n_conv_up_1 // 2, stride=1, kernel_size=9, padding=4, dilation=1),
                nn.BatchNorm2d(n_conv_up_1 // 2),
                activation,

                nn.Conv2d(n_conv_up_1 // 2, 1, stride=1, kernel_size=9, padding=4, dilation=1),
            )
            # todo, update if more layers are ... added!
            nn.init.xavier_normal_(self.conv_text_mask_out[0].weight)
            nn.init.constant_(self.conv_text_mask_out[0].bias, 0.0)
            nn.init.xavier_normal_(self.conv_text_mask_out[3].weight)
            nn.init.constant_(self.conv_text_mask_out[3].bias, 0.0)
            nn.init.xavier_normal_(self.conv_text_mask_out[6].weight)
            nn.init.constant_(self.conv_text_mask_out[6].bias, 0.0)
            nn.init.xavier_normal_(self.conv_text_mask_out[9].weight)
            nn.init.constant_(self.conv_text_mask_out[9].bias, 0.0)
            nn.init.xavier_normal_(self.conv_text_mask_out[12].weight)
            nn.init.constant_(self.conv_text_mask_out[12].bias, 0.0)
            """

    def reset_main_branches(self, in_channels, config):
        n_convs_up_1 = config.get("LectureNet.Network.Filters.ConvUp_1", 16)

        n_pmaps_1 = config.get("LectureNet.Network.Filters.PixelFeatures_1", 32)
        n_pmaps_2 = config.get("LectureNet.Network.Filters.PixelFeatures_2", 16)

        pix_kernel_size = config.get("LectureNet.Network.Filters.PixelKernel", 3)

        activation = FCN_LectureNet.GetActivation(config)

        self.set_main_branches(in_channels, pix_kernel_size, n_convs_up_1, n_pmaps_1, n_pmaps_2, activation)

    def get_batch_mid_block_features(self, batch_img):
        """
        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            x_conv1 = self.conv_block_pool_1(self.conv_down_block_1(batch_img))
            x_conv2 = self.conv_block_pool_2(self.conv_down_block_2(x_conv1))
            x_conv3 = self.conv_block_pool_3(self.conv_down_block_3(x_conv2))
            x_conv4 = self.conv_block_pool_4(self.conv_down_block_4(x_conv3))
            x_conv5 = self.conv_block_pool_5(self.conv_down_block_5(x_conv4))

            x_mid = self.mid_block(x_conv5)

        if using_cuda:
            x_mid = x_mid.cpu()

        x_mid = x_mid.numpy()

        return x_mid
        """
        raise Exception("This function has not been properly re-factored")

    def get_mid_block_features(self, PIL_image):
        """
        # prepare ...
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            x_conv1 = self.conv_block_pool_1(self.conv_down_block_1(batch_img))
            x_conv2 = self.conv_block_pool_2(self.conv_down_block_2(x_conv1))
            x_conv3 = self.conv_block_pool_3(self.conv_down_block_3(x_conv2))
            x_conv4 = self.conv_block_pool_4(self.conv_down_block_4(x_conv3))
            x_conv5 = self.conv_block_pool_5(self.conv_down_block_5(x_conv4))

            x_mid = self.mid_block(x_conv5)

        if using_cuda:
            x_mid = x_mid.cpu()

        x_mid = x_mid[0].numpy()

        return x_mid
        """
        raise Exception("This function has not been properly re-factored")

    def encode_decode(self, x0):
        encoder_features = self.encoder(x0)
        x_up1 = self.decoder(x0, encoder_features)

        return x_up1

    def get_batch_diff_images(self, batch_img, concat_features, downsample=None):
        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            x_up1 = self.encode_decode(batch_img)

            text_mask = self.conv_text_mask_out(x_up1)
            bin_text_mask = torch.sigmoid(text_mask)

            # using reconstruction branch ...
            rec_img = self.conv_reconstruct(x_up1)
            diff_img = (batch_img - rec_img) * bin_text_mask

            if concat_features:
                diff_img = torch.cat((diff_img, x_up1), 1)

            if downsample is not None:
                diff_img = nn.functional.max_pool2d(diff_img, kernel_size=downsample)

        if using_cuda:
            diff_img = diff_img.cpu()

        diff_img = diff_img.numpy()

        return diff_img

    def get_diff_image(self, PIL_image, concat_features, downsample=None):
        # prepare image and make it into batch
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        diff_img = self.get_batch_diff_images(batch_img, concat_features, downsample)

        diff_img = diff_img[0]

        return diff_img

    def forward(self, x0):
        x_up1 = self.encode_decode(x0)

        if not self.reconstruction_mode:
            # using text mask branch
            # Second Branch: Text Mask
            text_mask = self.conv_text_mask_out(x_up1)

            bin_text_mask = torch.sigmoid(text_mask)

            # using reconstruction branch ...
            # TODO:
            # rec_img = self.conv_reconstruct(x_up1) * 2.75
            rec_img = self.conv_reconstruct(x_up1)
            diff_img = (x0 - rec_img) * bin_text_mask

            # First Branch: Binary Image
            output = self.conv_binarizer(diff_img, x_up1)

            # return output, text_mask
            return output, text_mask, rec_img
        else:
            # using reconstruction mode ... output from reconstruction branch
            raw_img = self.conv_reconstruct(x_up1)

            # For Tanh ... simply produce output in range -1 to 1
            output = raw_img

            return output

    def binarize(self, PIL_image, return_others=False, force_binary=False, binary_treshold=128, apply_sigmoid=True):
        o_width, o_height = PIL_image.size
        width = o_width
        height = o_height
        # check for images bigger than 2.5 Mega Pixels
        while width * height > 2500000:
            PIL_image = PIL_image.resize((int(width / 2), int(height / 2)), PIL.Image.LANCZOS)
            width, height = PIL_image.size

        # make sure that each size is a multiple of ...32!
        divisor = 2 ** 5 #
        pre_pad_w, pre_pad_h = PIL_image.size
        padded_w = math.ceil(pre_pad_w / divisor) * divisor
        padded_h = math.ceil(pre_pad_h / divisor) * divisor

        pad_w = padded_w - pre_pad_w
        pad_h = padded_h - pre_pad_h
        padding = (0, 0, pad_w, pad_h)
        PIL_image = ImageOps.expand(PIL_image, border=padding, fill=0)

        # assert isinstance(PIL_image, Image)
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        device = next(self.parameters()).device
        batch_img = batch_img.to(device)

        with torch.no_grad():
            # res, text_mask = self.forward(batch_img)
            res, text_mask, rec_img = self.forward(batch_img)

            # applying sigmoid
            if apply_sigmoid:
                res = torch.sigmoid(res)
                text_mask = torch.sigmoid(text_mask)

        if res.is_cuda:
            res = res.cpu()
            text_mask = text_mask.cpu()
            rec_img = rec_img.cpu()

        binary = res[0, 0].numpy() * 255
        binary = binary.astype(np.uint8)

        if force_binary:
            # use hard threshold
            binary[binary >= binary_treshold] = 255
            binary[binary < binary_treshold] = 0

        if return_others:
            text_mask = text_mask[0, 0].numpy() * 255
            text_mask = text_mask.astype(np.uint8)

            if force_binary:
                # use hard threshold
                text_mask[text_mask >= binary_treshold] = 255
                text_mask[text_mask < binary_treshold] = 0

            rec_img = rec_img[0].numpy()
            rec_img = self.from_img_space_to_cv2(rec_img)

        # undo the padding first ...
        binary = binary[:pre_pad_h, :pre_pad_w]
        if return_others:
            text_mask = text_mask[:pre_pad_h, :pre_pad_w]
            rec_img = rec_img[:pre_pad_h, :pre_pad_w]

        if o_width != width:
            # need to resize them again ....
            if force_binary:
                binary = cv2.resize(binary, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
            else:
                binary = cv2.resize(binary, (o_width, o_height), interpolation=cv2.INTER_CUBIC)

            if return_others:
                if force_binary:
                    text_mask = cv2.resize(text_mask, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
                else:
                    text_mask = cv2.resize(text_mask, (o_width, o_height), interpolation=cv2.INTER_CUBIC)

                rec_img = cv2.resize(rec_img, (o_width, o_height), interpolation=cv2.INTER_NEAREST)

        """
        cv2.imshow("check", binary)
        if return_mask:
            cv2.imshow("mask", text_mask)
        cv2.waitKey()
        """
        if return_others:
            return binary, text_mask, rec_img
        else:
            return binary

    def from_img_space_to_cv2_sigmoid(self, image):
        # assume input is normalized numpy array in the same format as it is used for input images

        # from PIL to CV2 ... transpose
        image = np.transpose(image, (1, 2, 0))

        image[:, :, 0] *= 0.229
        image[:, :, 1] *= 0.224
        image[:, :, 2] *= 0.225

        image[:, :, 0] += 0.485
        image[:, :, 1] += 0.456
        image[:, :, 2] += 0.406

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    def from_img_space_to_cv2(self, image):
        # for tanh
        # tranpose .. from (Channels, H, W) to (H, W, Channels)
        image = np.transpose(image, (1, 2, 0))
        # scale from [-1, 1] to [-0.5, 0.5]
        image *= 0.5
        # translate from [-0.5, 0.5] to [0.0, 1.0]
        image += 0.5

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        # these should not be needed for Tanh
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    def from_img_space_to_cv2_scaled(self, image):
        # for tanh
        # tranpose .. from (Channels, H, W) to (H, W, Channels)
        image = np.transpose(image, (1, 2, 0))
        # scale from [-2.75, 2.75] to range unormalized range
        image[:, :, 0] *= 0.229
        image[:, :, 1] *= 0.224
        image[:, :, 2] *= 0.225
        # add mean color (should now be between 0.0 to 1.0)
        image[:, :, 0] += 0.485
        image[:, :, 1] += 0.456
        image[:, :, 2] += 0.406

        # swap channels
        tempo = image[:, :, 0].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = tempo

        # convert to uint8
        image *= 255
        # these should not be needed for Tanh
        image[image > 255] = 255
        image[image < 0] = 0
        image = image.astype(np.uint8)

        return image

    def reconstruct(self, PIL_image):
        # assert isinstance(PIL_image, Image)
        batch_img = FCN_LectureNet.prepare_image(PIL_image)

        using_cuda = next(self.parameters()).is_cuda

        if using_cuda:
            batch_img = batch_img.cuda(0)

        with torch.no_grad():
            res = self.forward(batch_img)

        if using_cuda:
            res = res.cpu()

        # De-normalize ...
        image = res[0].numpy()

        # for tanh ...
        image = self.from_img_space_to_cv2(image)

        return image

    @staticmethod
    def prepare_image(PIL_image):
        img_t = TF.to_tensor(PIL_image)
        # ... normalize the RGB values
        # img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # ... simply put RGB values in same range as Tanh
        img_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img_t = torch.unsqueeze(img_t, 0)

        return img_t

    @staticmethod
    def GetActivation(config):
        if config.contains("LectureNet.Network.Activation"):
            activation_type = config.get("LectureNet.Network.Activation.Type").lower()
            if activation_type == "gelu":
                return nn.GELU()
            elif activation_type == "silu":
                return nn.SiLU()
            elif activation_type == "relu":
                return nn.ReLU()
            elif activation_type == "leakyrelu":
                slope = config.get_float("LectureNet.Network.Activation.Parameters.Slope")
                return nn.LeakyReLU(slope)
            elif activation_type == "sigmoid":
                return nn.Sigmoid()
            elif activation_type == "tanh":
                return nn.Tanh()
            else:
                raise Exception(f"Activation {activation_type} currently not supported!")
        else:
            # Default activation, as used in the original LectureNet paper
            return nn.GELU()

    @staticmethod
    def CreateFromConfig(config, in_channels, reconstruction_mode, original_arch=True):
        n_convs_down_1 = config.get("LectureNet.Network.Filters.ConvDown_1", 16)
        n_convs_down_2 = config.get("LectureNet.Network.Filters.ConvDown_2", 32)
        n_convs_down_3 = config.get("LectureNet.Network.Filters.ConvDown_3", 64)
        n_convs_down_4 = config.get("LectureNet.Network.Filters.ConvDown_4", 128)
        n_convs_down_5 = config.get("LectureNet.Network.Filters.ConvDown_5", 256)

        n_convs_mid = config.get("LectureNet.Network.Filters.Middle", 512)

        n_upscale_5 = config.get("LectureNet.Network.Filters.Upsample_5", 256)
        n_convs_up_5 = config.get("LectureNet.Network.Filters.ConvUp_5", 256)

        n_upscale_4 = config.get("LectureNet.Network.Filters.Upsample_4", 128)
        n_convs_up_4 = config.get("LectureNet.Network.Filters.ConvUp_4", 128)

        n_upscale_3 = config.get("LectureNet.Network.Filters.Upsample_3", 64)
        n_convs_up_3 = config.get("LectureNet.Network.Filters.ConvUp_3", 64)

        n_upscale_2 = config.get("LectureNet.Network.Filters.Upsample_2", 32)
        n_convs_up_2 = config.get("LectureNet.Network.Filters.ConvUp_2", 32)

        n_upscale_1 = config.get("LectureNet.Network.Filters.Upsample_1", 16)
        n_convs_up_1 = config.get("LectureNet.Network.Filters.ConvUp_1", 16)

        n_pix_feats_1 = config.get("LectureNet.Network.Filters.PixelFeatures_1", 32)
        n_pix_feats_2 = config.get("LectureNet.Network.Filters.PixelFeatures_2", 16)

        pix_kernel_size = config.get("LectureNet.Network.Filters.PixelKernel", 3)

        kernel_size = config.get("LectureNet.Network.Filters.MainKernel", 3)

        activation = FCN_LectureNet.GetActivation(config)

        lecture_net = FCN_LectureNet(in_channels, n_convs_down_1, n_convs_down_2, n_convs_down_3, n_convs_down_4,
                                     n_convs_down_5, n_convs_mid,
                                     n_upscale_5, n_convs_up_5, n_upscale_4, n_convs_up_4, n_upscale_3, n_convs_up_3,
                                     n_upscale_2, n_convs_up_2, n_upscale_1, n_convs_up_1, kernel_size,
                                     n_pix_feats_1, n_pix_feats_2, pix_kernel_size, activation, reconstruction_mode,
                                     original_arch)

        return lecture_net
