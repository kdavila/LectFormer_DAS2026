from torch import nn as nn
from torch.nn import functional as F

from LM_Tools.configuration.configuration import Configuration
from .res_conv_block import ResConvBlock
from .util import ConvNormalizationHelper, compile_disable_if_windows, config_get_activation


class FCN_Encoder(nn.Module):
    def __init__(self, channels, kernel_size, activation, n_conv_down_1, n_conv_down_2, n_conv_down_3, n_conv_down_4,
                 n_conv_down_5, use_residuals=False, normalization=None):
        super(FCN_Encoder, self).__init__()

        assert isinstance(normalization, ConvNormalizationHelper)

        self.in_channels = channels
        self.n_conv_down_1 = n_conv_down_1
        self.n_conv_down_2 = n_conv_down_2
        self.n_conv_down_3 = n_conv_down_3
        self.n_conv_down_4 = n_conv_down_4
        self.n_conv_down_5 = n_conv_down_5

        # initial convolutions ...
        padding = int((kernel_size - 1) / 2)
        self.use_residuals = use_residuals
        if self.use_residuals:
            self.conv_down_block_1 = nn.Sequential(
                nn.Conv2d(channels, n_conv_down_1, stride=1, kernel_size=kernel_size, padding=padding),
                normalization.get_layer(n_conv_down_1),
                activation,
                ResConvBlock(n_conv_down_1, n_conv_down_1, kernel_size, activation, normalization)
            )
            # residual blocks ... just one block for now ...
            self.conv_down_block_2 = ResConvBlock(n_conv_down_1, n_conv_down_2, kernel_size, activation, normalization)
            self.conv_down_block_3 = ResConvBlock(n_conv_down_2, n_conv_down_3, kernel_size, activation, normalization)
            self.conv_down_block_4 = ResConvBlock(n_conv_down_3, n_conv_down_4, kernel_size, activation, normalization)
            self.conv_down_block_5 = ResConvBlock(n_conv_down_4, n_conv_down_5, kernel_size, activation, normalization)
        else:
            # Standard Encoder ....
            self.conv_down_block_1 = self.__simple_conv_block(channels, n_conv_down_1, kernel_size, activation, normalization)
            self.conv_down_block_2 = self.__simple_conv_block(n_conv_down_1, n_conv_down_2, kernel_size, activation, normalization)
            self.conv_down_block_3 = self.__simple_conv_block(n_conv_down_2, n_conv_down_3, kernel_size, activation, normalization)
            self.conv_down_block_4 = self.__simple_conv_block(n_conv_down_3, n_conv_down_4, kernel_size, activation, normalization)
            self.conv_down_block_5 = self.__simple_conv_block(n_conv_down_4, n_conv_down_5, kernel_size, activation, normalization)

        """
        self.conv_block_pool_1 = nn.MaxPool2d(2, return_indices=False)
        self.conv_block_pool_2 = nn.MaxPool2d(2, return_indices=False)
        self.conv_block_pool_3 = nn.MaxPool2d(2, return_indices=False)
        self.conv_block_pool_4 = nn.MaxPool2d(2, return_indices=False)
        self.conv_block_pool_5 = nn.MaxPool2d(2, return_indices=False)
        """

        self.__output_maps = n_conv_down_5

        # initializing weights
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def output_maps(self):
        return self.__output_maps

    def __simple_conv_block(self, n_conv_in, n_conv_out, kernel_size, activation, normalization):
        padding = int((kernel_size - 1) / 2)

        first_norm = normalization.get_layer(n_conv_out)
        second_norm = normalization.get_layer(n_conv_out)

        return nn.Sequential(
            nn.Conv2d(n_conv_in, n_conv_out, stride=1, kernel_size=kernel_size, padding=padding),
            first_norm,
            activation,
            nn.Conv2d(n_conv_out, n_conv_out, stride=1, kernel_size=kernel_size, padding=padding),
            second_norm,
            activation
        )

    @compile_disable_if_windows
    def _maybe_eager_pool(self, x):
        return F.max_pool2d(x, 2)

    def forward(self, x0):
        x_conv1_pre = self.conv_down_block_1(x0)
        # x_conv1 = self.conv_block_pool_1(x_conv1_pre)
        x_conv1 = self._maybe_eager_pool(x_conv1_pre)

        x_conv2_pre = self.conv_down_block_2(x_conv1)
        # x_conv2 = self.conv_block_pool_2(x_conv2_pre)
        x_conv2 = self._maybe_eager_pool(x_conv2_pre)

        x_conv3_pre = self.conv_down_block_3(x_conv2)
        # x_conv3 = self.conv_block_pool_3(x_conv3_pre)
        x_conv3 = self._maybe_eager_pool(x_conv3_pre)

        x_conv4_pre = self.conv_down_block_4(x_conv3)
        # x_conv4 = self.conv_block_pool_4(x_conv4_pre)
        x_conv4 = self._maybe_eager_pool(x_conv4_pre)

        x_conv5_pre = self.conv_down_block_5(x_conv4)
        # x_conv5 = self.conv_block_pool_5(x_conv5_pre)
        x_conv5 = self._maybe_eager_pool(x_conv5_pre)

        return [
            (x_conv1_pre, x_conv1),
            (x_conv2_pre, x_conv2),
            (x_conv3_pre, x_conv3),
            (x_conv4_pre, x_conv4),
            (x_conv5_pre, x_conv5)
        ]

    @staticmethod
    def CreateFromConfig(config, in_channels):
        isinstance(config, Configuration)
        n_convs_down_1 = config.get("LectureNet.Network.Filters.ConvDown_1", 16)
        n_convs_down_2 = config.get("LectureNet.Network.Filters.ConvDown_2", 32)
        n_convs_down_3 = config.get("LectureNet.Network.Filters.ConvDown_3", 64)
        n_convs_down_4 = config.get("LectureNet.Network.Filters.ConvDown_4", 128)
        n_convs_down_5 = config.get("LectureNet.Network.Filters.ConvDown_5", 256)

        use_residual = config.get("LectureNet.Network.Filters.EncoderResiduals", True)

        kernel_size = config.get("LectureNet.Network.Filters.MainKernel", 3)
        activation = config_get_activation(config)

        norm_config = config.get_subconfig("LectureNet.Network.Normalization")
        normalization = ConvNormalizationHelper.FromConfig(norm_config)

        return FCN_Encoder(in_channels, kernel_size, activation, n_convs_down_1, n_convs_down_2, n_convs_down_3,
                           n_convs_down_4, n_convs_down_5, use_residual, normalization)
