
import torch
import torch.nn as nn

from .util import ConvNormalizationHelper


class ResConvBlock(nn.Module):
    def __init__(self, n_conv_in, n_conv_out, kernel_size, activation, normalization):
        super(ResConvBlock, self).__init__()

        assert isinstance(normalization, ConvNormalizationHelper)

        padding = int((kernel_size - 1) / 2)

        first_norm = normalization.get_layer(n_conv_out)
        second_norm = normalization.get_layer(n_conv_out)
        if n_conv_in != n_conv_out:
            skip_norm = normalization.get_layer(n_conv_out)
        else:
            skip_norm = None

        self.__conv1 = nn.Conv2d(n_conv_in, n_conv_out, stride=1, kernel_size=kernel_size, padding=padding)
        self.__bn1 = first_norm
        self.__activation = activation

        self.__conv2 = nn.Conv2d(n_conv_out, n_conv_out, stride=1, kernel_size=kernel_size, padding=padding)
        self.__bn2 = second_norm

        if n_conv_in != n_conv_out:
            # a simple, linear transformation from in-conv-maps to out-conv-maps
            self.__skip_transformation = nn.Sequential(
                nn.Conv2d(n_conv_in, n_conv_out, stride=1, kernel_size=1),
                skip_norm
            )
        else:
            # not needed
            self.__skip_transformation = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x0):
        # either linear transformation o no transformation at all
        identity = self.__skip_transformation(x0)

        x1 = self.__conv1(x0)
        x1 = self.__bn1(x1)
        x1 = self.__activation(x1)

        x2 = self.__conv2(x1)
        x2 = self.__bn1(x2)
        x_out = self.__activation(identity + x2)

        return x_out
