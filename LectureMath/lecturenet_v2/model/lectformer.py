import math

import cv2
import numpy as np

import PIL
import PIL.Image
from PIL import ImageOps

import torch
import torch.nn as nn

import torchvision.transforms.functional as TF

from LM_Tools.configuration.configuration import Configuration

from .TransSkips import FCN_TransSkips
from .binarizers import FCN_BinarizerBranch, FCN_BinarizerHybridFusionBranch
from .decoder import FCN_Decoder
from .encoder import FCN_Encoder
from .util import config_get_activation, ConvNormalizationHelper
from .res_conv_block import ResConvBlock


class LectFormerAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, use_middle_block, mid_block, kernel_size, rec_kernel_size, activation,
                 tf_skips, normalization, prc_masked_features, laplacian_pyr_levels=None, text_det_mode=False,
                 n_bmaps_1=None):
        super(LectFormerAutoEncoder, self).__init__()

        assert isinstance(encoder, FCN_Encoder)
        assert isinstance(decoder, FCN_Decoder)

        assert isinstance(normalization, ConvNormalizationHelper)

        # initial convolutions ...
        padding = int((kernel_size - 1) / 2)

        self.__prc_masked_features = prc_masked_features

        if self.__prc_masked_features > 0:
            self.__up_2x = nn.Upsample(scale_factor=(2, 2), mode='nearest')
        else:
            self.__up_2x = None

        self.encoder = encoder

        mid_first_norm = normalization.get_layer(mid_block)
        mid_second_norm = normalization.get_layer(mid_block)

        self.__use_middle_block = use_middle_block
        if self.__use_middle_block:
            # using a middle conv. block
            self.mid_block = nn.Sequential(
                nn.Conv2d(self.encoder.n_conv_down_5, mid_block, stride=1, kernel_size=kernel_size, padding=padding),
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
        else:
            # not using a middle block ...
            self.mid_block = None

        self.skips = tf_skips

        self.decoder = decoder

        rec_padding = int((rec_kernel_size - 1) / 2)

        self._text_det_mode = text_det_mode

        if self._text_det_mode:
            self.conv_reconstruct = nn.Sequential(
                ResConvBlock(self.decoder.n_conv_up_1, n_bmaps_1, kernel_size, activation, normalization),
                nn.Conv2d(n_bmaps_1, 1, stride=1, kernel_size=5, padding=2),
            )
            self.conv_reconstruct[0].init_weights()
            nn.init.xavier_normal_(self.conv_reconstruct[1].weight)
            nn.init.constant_(self.conv_reconstruct[1].bias, 0.0)
        else:
            if laplacian_pyr_levels is None:
                out_channels = 3
            else:
                out_channels = laplacian_pyr_levels * 3

            self._laplacian_levels = laplacian_pyr_levels

            self.conv_reconstruct = nn.Sequential(
                nn.Conv2d(self.decoder.n_conv_up_1, out_channels, stride=1, kernel_size=rec_kernel_size, padding=rec_padding),
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

        _, x_conv5 = enc_feats[4]

        # middle block ...
        if self.__use_middle_block:
            x_mid = self.mid_block(x_conv5)
        else:
            # bypass
            x_mid = x_conv5

        if self.skips is not None:
            # using transformer skips
            # takes the masked inputs, and produces the outputs...
            dec_skips = self.skips(enc_feats)
            x_up1 = self.decoder((x_mid, dec_skips, x0))
        else:
            # not using transformer skips ...
            # decoder ...
            x_up1 = self.decoder((x_mid, enc_feats, x0))

        # final reconstruction branch
        raw_img = self.conv_reconstruct(x_up1)

        return raw_img

    def reconstruct(self, PIL_image, force_binary=False, binary_threshold=128):
        # make sure that each size is a multiple of ...32!
        pre_pad_w, pre_pad_h = PIL_image.size
        PIL_image = LectFormer.pad_image_to_power_of_2(PIL_image, 5)

        # assert isinstance(PIL_image, Image)
        batch_img = LectFormer.prepare_image(PIL_image, next(self.parameters()).device)

        with torch.no_grad():
            res = self.forward(batch_img)

        if res.is_cuda:
            res = res.cpu()

        # remove padding if any ...
        res = res[:, :, :pre_pad_h, :pre_pad_w]

        if self._text_det_mode:
            res = LectFormer.from_logits_to_bin_cv2(res, force_binary, binary_threshold)
            image = res
        elif self._laplacian_levels is not None and self._laplacian_levels > 1:
            # regular reconstruction, but combine multiple maps into one single output ...
            final_rec = None
            for level_idx in range(self._laplacian_levels):
                level_map_s = level_idx * 3
                level_map_e = level_map_s + 3
                level_rec = res[:, level_map_s:level_map_e, :, :]

                if level_idx == 0:
                    final_rec = level_rec
                else:
                    # add the maps to get the final reconstruction
                    final_rec = final_rec + level_rec

            res = final_rec

            # De-normalize ...
            image = res[0].numpy()

            # for tanh ...
            image = LectFormer.from_img_space_to_cv2(image)
        else:
            # assume regular reconstruction ...
            image = res[0].numpy()

            # for tanh ...
            image = LectFormer.from_img_space_to_cv2(image)

        return image

    @staticmethod
    def CreateFromConfig(config, in_channels, use_middle_block, use_tf_skips, prc_feat_masking,
                         laplacian_pyr_levels=None, text_det_mode=False):
        assert isinstance(config, Configuration)

        encoder = FCN_Encoder.CreateFromConfig(config, in_channels)
        decoder = FCN_Decoder.CreateFromConfig(config, use_middle_block)

        n_convs_mid = config.get("LectureNet.Network.Filters.Middle", 512)

        pix_kernel_size = config.get("LectureNet.Network.Heads.PixelKernel", 3)
        kernel_size = config.get("LectureNet.Network.Filters.MainKernel", 3)

        if text_det_mode:
            n_bmaps_1 = config.get("LectureNet.Network.Heads.BranchFeatures_1", 16)
        else:
            n_bmaps_1 = None

        norm_config = config.get_subconfig("LectureNet.Network.Normalization")
        normalization = ConvNormalizationHelper.FromConfig(norm_config)

        activation = config_get_activation(config)

        if use_tf_skips:
            fcn_skips = FCN_TransSkips.CreateFromConfig(config)
        else:
            fcn_skips = None

        auto_encoder = LectFormerAutoEncoder(
            encoder, decoder, use_middle_block, n_convs_mid, kernel_size, pix_kernel_size, activation, fcn_skips,
            normalization, prc_feat_masking, laplacian_pyr_levels, text_det_mode, n_bmaps_1
        )

        return auto_encoder


class LectFormer(nn.Module):
    def __init__(self, encoder, decoder, channels, fcn_skips, kernel_size, n_bmaps_1, n_pmaps_1, n_pmaps_2,
                 pixel_kernel_size, activation, normalization, fusion_type, detach_fusion,
                 disable_bg_head=False, disable_txt_head=False):
        super(LectFormer, self).__init__()

        assert isinstance(encoder, FCN_Encoder)
        assert isinstance(decoder, FCN_Decoder)
        assert isinstance(normalization, ConvNormalizationHelper)

        fusion_type = fusion_type.lower()
        valid_fusion_types = ["legacy", "none", "early_concat", "late_add", "hybrid", "aux_only",
                              "task_aux_only", "task_all", "task_all_extra"]
        if fusion_type not in valid_fusion_types:
            raise Exception(f"Invalid fusion type {fusion_type}, expecting: {valid_fusion_types}")

        if disable_bg_head and disable_txt_head and fusion_type != "none":
            fusion_type = "none"
            print(f"\nWARNING: Fusion Type {fusion_type} is invalid without auxiliary branches, falling back to none")

        if fusion_type in ["task_aux_only", "task_all", "task_all_extra"] and (disable_bg_head or disable_txt_head):
            raise Exception(f"Invalid Fusion Type {fusion_type} with bg and/or txt head disabled")

        self._fusion_type = fusion_type
        self._detach_fusion = detach_fusion

        # initial convolutions ...
        padding = int((kernel_size - 1) / 2)

        self.encoder = encoder

        assert isinstance(fcn_skips, FCN_TransSkips)
        self.skips = fcn_skips

        print(f"- Active Skips: {self.skips.active_skips}")
        self.decoder = decoder

        print(f"Residuals: Encoder={encoder.use_residuals}, Decoder={decoder.use_residual}")
        print(f"Normalization: {normalization}")

        self._disable_bg_head = disable_bg_head
        self._disable_txt_head = disable_txt_head

        # FIRST OUTPUT BRANCH: BINARIZATION
        self.conv_binarizer = None

        # SECOND OUTPUT BRANCH: Text Masks
        self.conv_text_mask_out = None

        self.set_main_branches(channels, pixel_kernel_size, self.decoder.n_conv_up_1, n_pmaps_1, n_pmaps_2, n_bmaps_1,
                               activation, normalization)

        if not self._disable_bg_head:
            # THIRD OUTPUT BRANCH: reconstruction
            self.conv_reconstruct = nn.Sequential(
                ResConvBlock(self.decoder.n_conv_up_1, n_bmaps_1, kernel_size, activation, normalization),
                nn.Conv2d(n_bmaps_1, 3, stride=1, kernel_size=kernel_size, padding=padding),
                nn.Tanh(),
            )
            self.conv_reconstruct[0].init_weights()
            nn.init.xavier_normal_(self.conv_reconstruct[1].weight)
            nn.init.constant_(self.conv_reconstruct[1].bias, 0.0)

        # self.reconstruction_mode = reconstruction_mode

    def is_bg_head_disabled(self):
        return self._disable_bg_head

    def is_txt_head_disabled(self):
        return self._disable_txt_head

    def set_main_branches(self, channels, kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2, n_bmaps_1, activation,
                          normalization):
        padding = int((kernel_size - 1) / 2)

        # FIRST OUTPUT BRANCH: Binarization

        if self._fusion_type == "legacy":
            # (32 -> 16 -> 3) with aggressive fusion!
            self.conv_binarizer = FCN_BinarizerBranch(channels, kernel_size, n_conv_up_1, n_pmaps_1, n_pmaps_2,
                                                      activation, normalization)
        else:
            early_concat = self._fusion_type in ["early_concat", "hybrid"]
            late_add = self._fusion_type in ["late_add", "hybrid"]
            aux_only = self._fusion_type == "aux_only"

            self.conv_binarizer = FCN_BinarizerHybridFusionBranch(channels, kernel_size, n_conv_up_1, n_pmaps_1,
                                                                  n_pmaps_2, activation, normalization, early_concat,
                                                                  late_add, aux_only, self._detach_fusion)

        if not self._disable_txt_head:
            self.conv_text_mask_out = nn.Sequential(
                ResConvBlock(n_conv_up_1, n_bmaps_1, kernel_size, activation, normalization),
                nn.Conv2d(n_bmaps_1, 1, stride=1, kernel_size=5, padding=2),
            )
            self.conv_text_mask_out[0].init_weights()
            nn.init.xavier_normal_(self.conv_text_mask_out[1].weight)
            nn.init.constant_(self.conv_text_mask_out[1].bias, 0.0)

    def encode_decode(self, x0):

        # encode ...
        enc_outs = self.encoder(x0)
        # last output ....
        x_conv5 = enc_outs[4][1]

        dec_skips = self.skips(enc_outs)

        # decoder ...
        x_up1 = self.decoder((x_conv5, dec_skips, x0))

        return x_up1

    def forward(self, x0):
        x_up1 = self.encode_decode(x0)

        if self._fusion_type in ["task_aux_only", "task_all", "task_all_extra"]:
            # reconstruction has 3 layers (because of Tanh activation)
            rec_feats = self.conv_reconstruct[0](x_up1)
            rec_img = self.conv_reconstruct[2](self.conv_reconstruct[1](rec_feats))

            # text detection only has 2 layers (returns logits)
            text_feats = self.conv_text_mask_out[0](x_up1)
            text_mask = self.conv_text_mask_out[1](text_feats)
            bin_text_mask = torch.sigmoid(text_mask)

            # 3 channels
            diff1 = x0 - rec_img
            # 3 channels
            diff2 = x0 * bin_text_mask
            # 3 channels
            diff3 = (x0 - rec_img) * bin_text_mask
            # 9 channels in total
            diff_full = torch.cat((diff1, diff2, diff3), dim=1)

            output = self.conv_binarizer(x_up1, rec_feats, text_feats, diff_full)
            return output, text_mask, rec_img
        else:
            if not self._disable_txt_head:
                # using text mask branch
                # Second Branch: Text Mask
                text_mask = self.conv_text_mask_out(x_up1)
                bin_text_mask = torch.sigmoid(text_mask)
            else:
                text_mask = None
                bin_text_mask = None

            if not self._disable_bg_head:
                # using reconstruction branch ...
                rec_img = self.conv_reconstruct(x_up1)
            else:
                rec_img = None

            if (not self._disable_txt_head) and (not self._disable_bg_head):
                # original design: use all branches
                diff_img = (x0 - rec_img) * bin_text_mask
            else:
                # at least one of the heads is disabled ...
                if not self._disable_txt_head:
                    # use the text estimation ...
                    diff_img = x0 * bin_text_mask
                elif not self._disable_bg_head:
                    # use bg estimation ...
                    diff_img = x0 - rec_img
                else:
                    # no auxiliaries ...
                    diff_img = x0 if self._fusion_type == "legacy" else None

            if self._fusion_type == "legacy" and self._detach_fusion:
                # for old design ... new design can do this internally ...
                diff_img = diff_img.detach()

            # First Branch: Binary Image
            output = self.conv_binarizer(diff_img, x_up1)

            # return output, text_mask
            return output, text_mask, rec_img

    def binarize(self, PIL_image, return_others=False, force_binary=False, binary_treshold=128):
        o_width, o_height = PIL_image.size
        width, height = o_width, o_height
        # check for images bigger than 2.5 Mega Pixels
        while width * height > 2500000:
            PIL_image = PIL_image.resize((int(width / 2), int(height / 2)), PIL.Image.LANCZOS)
            width, height = PIL_image.size

        # make sure that each size is a multiple of ...32!
        pre_pad_w, pre_pad_h = PIL_image.size
        PIL_image = LectFormer.pad_image_to_power_of_2(PIL_image, 5)

        # assert isinstance(PIL_image, Image)
        batch_img = LectFormer.prepare_image(PIL_image, next(self.parameters()).device)

        with torch.no_grad():
            # res, text_mask = self.forward(batch_img)
            binary, text_mask, rec_img = self.forward(batch_img)

        if binary.is_cuda:
            binary = binary.cpu()
            if text_mask is not None:
                text_mask = text_mask.cpu()
            if rec_img is not None:
                rec_img = rec_img.cpu()

        # applying sigmoid
        binary = LectFormer.from_logits_to_bin_cv2(binary, force_binary, binary_treshold)

        if return_others:
            if text_mask is not None:
                text_mask = LectFormer.from_logits_to_bin_cv2(text_mask, force_binary, binary_treshold)

            if rec_img is not None:
                rec_img = rec_img[0].numpy()
                rec_img = LectFormer.from_img_space_to_cv2(rec_img)

        # undo the padding first ...
        binary = binary[:pre_pad_h, :pre_pad_w]
        if return_others:
            if text_mask is not None:
                text_mask = text_mask[:pre_pad_h, :pre_pad_w]
            if rec_img is not None:
                rec_img = rec_img[:pre_pad_h, :pre_pad_w]

        if o_width != width:
            # need to resize them again ....
            if force_binary:
                binary = cv2.resize(binary, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
            else:
                binary = cv2.resize(binary, (o_width, o_height), interpolation=cv2.INTER_CUBIC)

            if return_others:
                if text_mask is not None:
                    if force_binary:
                        text_mask = cv2.resize(text_mask, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
                    else:
                        text_mask = cv2.resize(text_mask, (o_width, o_height), interpolation=cv2.INTER_CUBIC)
                if rec_img is not None:
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

    def reconstruct(self, PIL_image):
        # assert isinstance(PIL_image, Image)
        batch_img = LectFormer.prepare_image(PIL_image, next(self.parameters()).is_cuda)

        with torch.no_grad():
            _, _, res = self.forward(batch_img)

        if res.is_cuda:
            res = res.cpu()

        image = res[0].numpy()
        # for tanh ...
        image = LectFormer.from_img_space_to_cv2(image)

        return image

    @staticmethod
    def from_logits_to_bin_cv2(t_binary, force_binary, binary_t):
        # first, convert logits to actual 0-1 values
        t_binary = torch.sigmoid(t_binary)
        # from (B, C, H, W) on [0-1] in Torch to (H, W) on [0-255] in Numpy
        text_mask = t_binary[0, 0].numpy() * 255
        text_mask = text_mask.astype(np.uint8)
        if force_binary:
            # use hard threshold
            text_mask[text_mask >= binary_t] = 255
            text_mask[text_mask < binary_t] = 0

        return text_mask

    @staticmethod
    def from_img_space_to_cv2(image):
        # for outputs produced with Tanh
        # (Channels, H, W) to (H, W, Channels)
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
    def pad_image_to_power_of_2(PIL_image, exponent=5):
        divisor = 2 ** exponent
        pre_pad_w, pre_pad_h = PIL_image.size
        padded_w = math.ceil(pre_pad_w / divisor) * divisor
        padded_h = math.ceil(pre_pad_h / divisor) * divisor

        pad_w = padded_w - pre_pad_w
        pad_h = padded_h - pre_pad_h
        padding = (0, 0, pad_w, pad_h)
        return ImageOps.expand(PIL_image, border=padding, fill=0)

    @staticmethod
    def prepare_image(PIL_image, device):
        img_t = TF.to_tensor(PIL_image)
        # ... normalize the RGB values
        # img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # ... simply put RGB values in same range as Tanh
        img_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        img_t = torch.unsqueeze(img_t, 0)

        img_t = img_t.to(device)

        return img_t

    @staticmethod
    def CreateFromConfig(config, in_channels):
        assert isinstance(config, Configuration)

        encoder = FCN_Encoder.CreateFromConfig(config, in_channels)
        decoder = FCN_Decoder.CreateFromConfig(config, False)

        n_bmaps_1 = config.get("LectureNet.Network.Heads.BranchFeatures_1", 16)

        n_pix_feats_1 = config.get("LectureNet.Network.Heads.PixelFeatures_1", 32)
        n_pix_feats_2 = config.get("LectureNet.Network.Heads.PixelFeatures_2", 16)

        pix_kernel_size = config.get("LectureNet.Network.Heads.PixelKernel", 3)

        disable_txt_head = not config.get("LectureNet.Network.Heads.TextMaskHeadEnabled", True)
        disable_bg_head = not config.get("LectureNet.Network.Heads.BackgroundHeadEnabled", True)
        if disable_txt_head:
            print("- Text Mask Head has been disabled!")
        if disable_bg_head:
            print("- Text Mask Head has been disabled!")

        fusion_type = config.get("LectureNet.Network.Heads.Fusion", "Invalid")
        detach_fusion = config.get("LectureNet.Network.Heads.DetachAuxiliary", False)

        kernel_size = config.get("LectureNet.Network.Filters.MainKernel", 3)

        norm_config = config.get_subconfig("LectureNet.Network.Normalization")
        normalization = ConvNormalizationHelper.FromConfig(norm_config)

        activation = config_get_activation(config)

        fcn_skips = FCN_TransSkips.CreateFromConfig(config)

        lecture_net = LectFormer(
            encoder, decoder, in_channels, fcn_skips, kernel_size, n_bmaps_1, n_pix_feats_1, n_pix_feats_2,
            pix_kernel_size, activation, normalization, fusion_type, detach_fusion,
            disable_bg_head, disable_txt_head
        )

        return lecture_net
