import torch
from torch import nn as nn

from LectureMath.lecturenet_v2.model.res_conv_block import ResConvBlock
from LectureMath.lecturenet_v2.model.util import ConvNormalizationHelper


class FCN_BinarizerBranch(nn.Module):
    def __init__(self, channels, kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2, activation, normalization):
        super(FCN_BinarizerBranch, self).__init__()

        print("-> Using FCN Binarizer - Original Aggressive Fusion")

        assert isinstance(normalization, ConvNormalizationHelper)

        padding = int((kernel_size - 1) / 2)

        first_norm = normalization.get_layer(n_pmaps_1)
        second_norm = normalization.get_layer(n_pmaps_2)

        # .... now use 1D convolutions ...
        inputs_conv_pixels_1 = channels + n_conv_up_1
        # inputs_conv_pixels_1 = channels + n_conv_up_1 + 1
        self.conv_pixels_1 = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_1, n_pmaps_1, stride=1, kernel_size=kernel_size, padding=padding),
            first_norm,
            activation
        )
        # TODO: this is not the same as other parts of the network!
        nn.init.xavier_normal_(self.conv_pixels_1[0].weight)
        nn.init.constant_(self.conv_pixels_1[0].bias, 0.0)

        inputs_conv_pixels_2 = channels + n_pmaps_1
        # inputs_conv_pixels_2 = channels + n_pmaps_1 + 1
        self.conv_pixels_2 = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_2, n_pmaps_2, stride=1, kernel_size=kernel_size, padding=padding),
            second_norm,
            activation
        )
        # TODO: this is not the same as other parts of the network!
        nn.init.xavier_normal_(self.conv_pixels_2[0].weight)
        nn.init.constant_(self.conv_pixels_2[0].bias, 0.0)

        # output 1: binary
        inputs_conv_pixels_3 = channels + n_pmaps_2
        # inputs_conv_pixels_3 = channels + n_pmaps_2 + 1
        self.conv_out = nn.Sequential(
            nn.Conv2d(inputs_conv_pixels_3, 1, stride=1, kernel_size=kernel_size, padding=padding),
        )
        # TODO: update if more layers are added!!
        # TODO: this is not the same as other parts of the network!
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

        # ... apply 1x1 convolution #2
        x_pixels_2 = self.conv_pixels_2(x_pixels_1)

        # ... add the input maps ...
        x_pixels_2 = torch.cat((diff_img, x_pixels_2), 1)

        # ... get last combination (NO SIGMOID)
        output = self.conv_out(x_pixels_2)

        return output


class FCN_BinarizerHybridFusionBranch(nn.Module):
    def __init__(self, aux_channels, kernel_size, n_conv_up_1, n_main_maps, n_aux_maps, activation, normalization,
                 early_concat=True, late_add=True, aux_only=False, detach_input=False):
        super(FCN_BinarizerHybridFusionBranch, self).__init__()

        print(f"-> Using FCN Binarizer with Hybrid Fusion (Early Concat = {early_concat}, Late Addition = {late_add})")

        assert isinstance(normalization, ConvNormalizationHelper)

        # padding = int((kernel_size - 1) / 2)
        self._early_concat = early_concat
        self._late_addition = late_add
        self._aux_only = aux_only
        self._detach_input = detach_input

        if not self._aux_only:
            main_in_channels = n_conv_up_1 + (aux_channels if self._early_concat else 0)
            self.conv_main_bin = nn.Sequential(
                ResConvBlock(main_in_channels, n_main_maps, kernel_size, activation, normalization),
                nn.Conv2d(n_main_maps, 1, stride=1, kernel_size=5, padding=2),
            )
            self.conv_main_bin[0].init_weights()
            nn.init.xavier_normal_(self.conv_main_bin[1].weight)
            nn.init.constant_(self.conv_main_bin[1].bias, 0.0)
        else:
            self.conv_main_bin = nn.Identity()

        if self._late_addition or self._aux_only:
            # using auxiliary late fusion ...
            self.conv_aux_bin = nn.Sequential(
                nn.Conv2d(aux_channels, n_aux_maps, stride=1, kernel_size=1, padding=0),
                activation,
                nn.Conv2d(n_aux_maps, 1, stride=1, kernel_size=1, padding=0),
            )
            nn.init.xavier_normal_(self.conv_aux_bin[0].weight)
            nn.init.constant_(self.conv_aux_bin[0].bias, 0.0)
            nn.init.xavier_normal_(self.conv_aux_bin[2].weight)
            nn.init.constant_(self.conv_aux_bin[2].bias, 0.0)
        else:
            # disable auxiliary late fusion ...
            self.conv_aux_bin = nn.Identity()

        # dynamic weight ....
        self.aux_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, diff_img, x_up1):
        if self._detach_input:
            # ensure it is detached ...
            diff_img = diff_img.detach()

        if self._early_concat:
            # concatenate with main features
            x_main = torch.cat((x_up1, diff_img), dim=1)
        else:
            # use backbone features only
            x_main = x_up1

        # this will be the identify if aux_only is enabled
        out_main = self.conv_main_bin(x_main)

        if self._late_addition or self._aux_only:
            # get auxiliary binarization
            out_aux = self.conv_aux_bin(diff_img)

            if self._aux_only:
                # use the auxiliary binarization directly ...
                return out_aux
            else:
                # ... fuse ....
                out_bin = out_main + self.aux_scale * out_aux

                return out_bin
        else:
            # disable auxiliary late fusion ...
            return out_main


