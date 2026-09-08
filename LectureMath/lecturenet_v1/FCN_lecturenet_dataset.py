
import random

import cv2
import numpy as np
from PIL import Image
import multiprocessing as mp

import torch
import torch.utils
import torch.utils.data
from networkx.linalg.spectrum import laplacian_spectrum

from torchvision import transforms
import torchvision.transforms.functional as TF

from LM_Tools.optimization.dataset_helper import DatasetHelper
from LM_Tools.optimization.data_augmentation_parameters import DataAugmentationParameters


def pyrup_to(img, target_hw):
    """Repeated pyrUp until reaching target size (exact via dstsize each step)."""
    out = img
    th, tw = target_hw.shape[:2]
    while out.shape[0] < th or out.shape[1] < tw:
        # Compute next expected size but clamp to target exactly
        nh = min(th, out.shape[0] * 2)
        nw = min(tw, out.shape[1] * 2)
        out = cv2.pyrUp(out, dstsize=(nw, nh))
    # In case of odd size quirks, force exact final shape
    if out.shape[0] != th or out.shape[1] != tw:
        out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_LINEAR)
    return out

def compute_laplacian_targets(input_img, pyramid_levels, norm_mean, norm_std):
    current_desc = input_img.astype(np.float32).copy()

    if norm_mean is not None:
        # first, make between 0-1
        current_desc /= 255.0
        # then apply normalization
        current_desc = (current_desc - norm_mean) / norm_std

    # first, get the Gaussian Pyramid ...
    gaussian = [current_desc]
    G = current_desc
    # cv2.imshow(f"G: -1", G.astype(np.uint8))
    for level_idx in range(pyramid_levels - 1):
        G = cv2.pyrDown(G)
        # cv2.imshow(f"G: {level_idx}", cv2.resize(G.astype(np.uint8), (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST))
        gaussian.append(G)

    # laplacian = []
    laplacian_stretched = []
    for i in range(pyramid_levels - 1):
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        L = gaussian[i] - up

        # L_vis = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow(f"L: {i}", cv2.resize(L_vis.astype(np.uint8) , (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST))
        # laplacian.append(L)

        laplacian_stretched.append(pyrup_to(L, input_img))

    # laplacian.append(gaussian[-1])  # smallest Gaussian residual
    laplacian_stretched.append(pyrup_to(gaussian[-1], input_img))
    # cv2.imshow(f"L: -1", cv2.resize(gaussian[-1].astype(np.uint8), (tempo_img.shape[1], tempo_img.shape[0]),
    #                                  cv2.INTER_NEAREST))
    """
    # sanity-check: is the reconstruction correct?
    # G = laplacian[-1]
    G_stretch = laplacian_stretched[-1]
    # cv2.imshow(f"Rec approx: {-1}", G_stretch.astype(np.uint8))
    # cv2.imshow(f"Rec: -1",
    #            cv2.resize(G.astype(np.uint8), (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST))
    for i in range(len(laplacian_stretched) - 2, -1, -1):
        # G = cv2.pyrUp(G, dstsize=(laplacian[i].shape[1], laplacian[i].shape[0]))
        # G = G + laplacian[i]
        # cv2.imshow(f"Rec proper: {i}",
        #            cv2.resize(G.astype(np.uint8), (input_img.shape[1], input_img.shape[0]), cv2.INTER_NEAREST))

        G_stretch += laplacian_stretched[i]

        vis_copy = G_stretch.copy()
        vis_copy[vis_copy < 0] = 0
        vis_copy[vis_copy > 255] = 255
        cv2.imshow(f"Rec approx: {i}", vis_copy.astype(np.uint8))

    cv2.waitKey()
    """
    return laplacian_stretched


class LectureNet_DataSet(torch.utils.data.Dataset):
    def __init__(self, image_list, ground_truth_list, reconstruction_mode,
                 crop_size=None, crop_remove_empty_borders=False, crop_min_fg_prc=None, crop_min_fg_ratio=None,
                 augmentation_params=None,
                 weight_expansion=None, weight_fg_extra=None,
                 text_region_masks_expansion=None,
                 reconstruct_type=None, reconstruct_filter_K=None, reconstruct_masked=False,
                 reconstruct_lap_pyramid=False, lap_pyramid_levels=5, invert_gt=False):

        if ground_truth_list is not None:
            assert len(image_list) == len(ground_truth_list)

        self.image_list = image_list
        self.ground_truth_list = ground_truth_list

        self.reconstruction_mode = reconstruction_mode

        self.crop_size = crop_size
        self.crop_remove_empty_borders = crop_remove_empty_borders
        # self.crop_min_fg_prc = crop_min_fg_prc
        # self.crop_min_fg_ratio = crop_min_fg_ratio
        # encoding "None" for the multi-processing wrapper..
        self._fg_sampling_lock = mp.RLock()
        if crop_min_fg_prc is None or crop_min_fg_prc < 0.0:
            crop_min_fg_prc = -1.0
        if crop_min_fg_ratio is None or crop_min_fg_ratio < 0.0:
            crop_min_fg_ratio = -1.0

        self._crop_min_fg_prc = mp.Value("d", float(crop_min_fg_prc), lock=False)
        self._crop_min_fg_ratio = mp.Value("d", float(crop_min_fg_ratio), lock=False)

        if augmentation_params is None:
            # empty augmentation parameters
            self.augmentation_params = DataAugmentationParameters()
        else:
            self.augmentation_params = augmentation_params

        self.weight_expansion = weight_expansion
        self.weight_fg_extra = weight_fg_extra
        if weight_expansion is None:
            self.weight_st_element = None
        else:
            disk_size = (self.weight_expansion * 2 + 1, self.weight_expansion * 2 + 1)
            self.weight_st_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, disk_size )

        self.text_region_masks_expansion = text_region_masks_expansion
        if text_region_masks_expansion is None or text_region_masks_expansion == 0:
            self.text_region_mask_st = None
        else:
            disk_size = (self.text_region_masks_expansion * 2 + 1, self.text_region_masks_expansion * 2 + 1)
            self.text_region_mask_st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, disk_size)

        self.invert_gt = invert_gt

        self.total_foreground = None
        self.total_background = None

        self.preloaded_images = None
        self.preloaded_ground_truths = None

        if reconstruct_type is not None:
            self.reconstruct_type = reconstruct_type.lower()
        else:
            self.reconstruct_type = "none"
        self.reconstruct_filter_K = reconstruct_filter_K
        self.reconstruct_masked = reconstruct_masked
        self.reconstruct_lap_pyramid = reconstruct_lap_pyramid

        self.lap_pyramid_levels = lap_pyramid_levels

        if self.reconstruct_type not in ["none", "raw", "mean", "median", "gaussian", "inpaint"]:
            raise Exception(f"Reconstruction type {reconstruct_type} not supported")

        if self.ground_truth_list is None and reconstruct_masked:
            raise Exception(f"Cannot do masked reconstruction without ground truth masks")

        if self.reconstruct_type == "inpaint" and not reconstruct_masked:
            raise Exception(f"Cannot do inpaint reconstruction without ground truth masks and masked mode")

    @staticmethod
    def _encode_optional(value):
        return -1.0 if value is None else float(value)

    @staticmethod
    def _decode_optional(value):
        return None if value < 0.0 else value

    def set_fg_sampling_params(self, fg_sampling_ratio, fg_min_prc):
        with self._fg_sampling_lock:
            self._crop_min_fg_ratio.value = self._encode_optional(fg_sampling_ratio)
            self._crop_min_fg_prc.value = self._encode_optional(fg_min_prc)

    def get_fg_sampling_params(self):
        with self._fg_sampling_lock:
            ratio = self._decode_optional(self._crop_min_fg_ratio.value)
            min_prc = self._decode_optional(self._crop_min_fg_prc.value)

        return ratio, min_prc

    def _prepare_image_pair(self, img, gt_img):
        # for later usage with PIL
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w, _ = img.shape

        if self.crop_remove_empty_borders and gt_img is not None:
            # find region that has text
            horizontal_range = np.nonzero((255 - gt_img).max(axis=0))[0]
            x_min_val = horizontal_range[0]
            x_max_val = horizontal_range[-1]

            vertical_range = np.nonzero((255 - gt_img).max(axis=1))[0]
            y_min_val = vertical_range[0]
            y_max_val = vertical_range[-1]

            half_w_margin = 10
            half_h_margin = 10

            start_x = max(0, x_min_val - half_w_margin)
            end_x = min(w, x_max_val + half_w_margin)

            start_y = max(0, y_min_val - half_h_margin)
            end_y = min(h, y_max_val + half_h_margin)

            # check ...
            if self.crop_size is not None:
                if end_x - start_x < self.crop_size[1]:
                    mid_point = int((start_x + end_x) / 2)
                    start_x = max(0, mid_point - int(self.crop_size[1] / 2 + 1))
                    end_x = min(w, start_x + self.crop_size[1])

                if end_y - start_y < self.crop_size[0]:
                    mid_point = int((start_y + end_y) / 2)
                    start_y = max(0, mid_point - int(self.crop_size[0] / 2 + 1))
                    end_y = min(h, start_y + self.crop_size[0])

            img = img[start_y:end_y, start_x:end_x]
            gt_img = gt_img[start_y:end_y, start_x:end_x]

            # update
            h, w, _ = img.shape

        # check for auto-resize (if the image is too small)
        if self.crop_size is not None and (h < self.crop_size[0] or w < self.crop_size[1]):
            # resize the image
            w_scale_factor = self.crop_size[1] / w
            h_scale_factor = self.crop_size[0] / h

            if w_scale_factor > h_scale_factor:
                # upscale by width
                new_height = int(round(h * w_scale_factor))
                new_width = self.crop_size[1]
            else:
                new_height = self.crop_size[0]
                new_width = int(round(w * h_scale_factor))

            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            if gt_img is not None:
                gt_img = cv2.resize(gt_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        return img, gt_img

    def load_image_pair(self, img_filename, gt_filename):
        img = cv2.imread(img_filename)

        if gt_filename is not None:
            gt_img = cv2.imread(gt_filename)
            # for later usage as GT
            gt_img = gt_img[:, :, 0]

            if self.invert_gt:
                # this whole class always assumes that FG=0, BG=255
                gt_img = 255 - gt_img
        else:
            gt_img = None

        img, gt_img = self._prepare_image_pair(img, gt_img)

        # cv2.imwrite("ZZZ.png", img)
        # cv2.waitKey()
        # x = 0 / 0
        return img, gt_img

    def preload(self, verbose=True):
        self.total_foreground = 0.0
        self.total_background = 0.0
        self.preloaded_images = []
        if self.ground_truth_list is not None:
            self.preloaded_ground_truths = []

        for img_idx, img_filename in enumerate(self.image_list):
            if verbose:
                print(" " * 120, end="\r")
                print("{0:d} - {1:s}".format(img_idx + 1, img_filename), end="\r")

            if self.ground_truth_list is not None:
                gt_filename = self.ground_truth_list[img_idx]
            else:
                gt_filename = None

            # read the image(s) directly as buffer(s) ...
            with open(img_filename, "rb") as f:
                encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
            if self.ground_truth_list is not None:
                with open(gt_filename, "rb") as f:
                    encoded_gt = np.frombuffer(f.read(), dtype=np.uint8)
            else:
                encoded_gt = None

            # img, gt_img = self.load_image_pair(img_filename, gt_filename)

            if self.ground_truth_list is not None:
                # TODO: is this worth fixing?
                # count background and foreground pixels
                # image_foreground = (gt_img == 0).sum()
                # self.total_foreground += image_foreground
                # self.total_background += (gt_img.size - image_foreground)
                pass

            # encode and store
            # ... main image ...
            # flag, encoded_img = cv2.imencode(".png", img)
            self.preloaded_images.append(encoded_img)
            # ... gt image ....
            if self.ground_truth_list is not None:
                # flag, encoded_gt = cv2.imencode(".png", gt_img)
                self.preloaded_ground_truths.append(encoded_gt)

        if verbose:
            print("\nImage pre-loading complete!")

    def __len__(self):
        return len(self.image_list)

    def get_full_image(self, index, with_gt=False):
        if self.preloaded_images is not None:
            # a compressed copy is stored on RAM ... decode and use!
            compresed_img = self.preloaded_images[index]
            raw_img = cv2.imdecode(compresed_img, cv2.IMREAD_COLOR)
            if with_gt:
                compressed_gt = self.preloaded_ground_truths[index]
                raw_gt = cv2.imdecode(compressed_gt, cv2.IMREAD_GRAYSCALE)

                if self.invert_gt:
                    # this whole class always assumes that FG=0, BG=255
                    raw_gt = 255 - raw_gt
            else:
                raw_gt = None

            # Note that earlier versions did this before caching
            # this version trades-off computing to save a lot of memory
            raw_img, raw_gt = self._prepare_image_pair(raw_img, raw_gt)

            pil_img = Image.fromarray(raw_img)

            if not with_gt:
                return pil_img
            else:
                pil_gt = Image.fromarray(raw_gt)

                return pil_img, pil_gt
        else:
            # load from Disk (slower!)
            img_filename = self.image_list[index]
            if self.ground_truth_list is not None:
                gt_filename = self.ground_truth_list[index]
            else:
                gt_filename = None

            raw_img, raw_gt = self.load_image_pair(img_filename, gt_filename)

            pil_img = Image.fromarray(raw_img)

            if not with_gt:
                return pil_img
            else:

                pil_gt = Image.fromarray(raw_gt)

                return pil_img, pil_gt

    def __getitem__(self, index):
        if self.ground_truth_list is not None:
            pil_img, pil_gt = self.get_full_image(index, True)
        else:
            pil_img = self.get_full_image(index, False)
            pil_gt = None

        if self.augmentation_params.flip_chance is not None:
            # try horizontal flipping
            if self.augmentation_params.should_apply_flip():
                pil_img = TF.hflip(pil_img)
                if pil_gt is not None:
                    pil_gt = TF.hflip(pil_gt)

            # try vertical flipping
            if self.augmentation_params.should_apply_flip():
                pil_img = TF.vflip(pil_img)
                if pil_gt is not None:
                    pil_gt = TF.vflip(pil_gt)

        # rotation and zoom
        zoom_value, rotation_value = self.augmentation_params.random_zoom_and_rotation_values()

        if zoom_value is not None or rotation_value is not None:
            # apply transformation to the image ...
            img_np = np.asarray(pil_img)
            # w, h
            target_size = (img_np.shape[1], img_np.shape[0])
            img_np = DatasetHelper.apply_scaling_rotation(target_size, img_np, rotation_value, zoom_value, (0, 0, 0))
            pil_img = Image.fromarray(img_np)
            if pil_gt is not None:
                gt_np = np.asarray(pil_gt)
                # TODO: There is a bug here, but it was intentionally left behind to match results on the DAS 2026 paper
                #       The issue is padding with 0, but GT for Stage (Lectures or Segmentation Datasets), will always
                #       have white (255) as the background color, provided that gt_invert is properly set up
                # TODO: Stage 2. GT is not inverted, but it represents text segmentation, not true binarization
                #       Padding with 0 is valid for datasets such as LSVT, where text detection mask is assumed to have
                #       black background and white foreground
                # TODO: conclusion. A config value or additional info is needed to know when to use 0 (Stage 2) or 255 (Stage 3)
                gt_np = DatasetHelper.apply_scaling_rotation(target_size, gt_np, rotation_value, zoom_value, 0)
                pil_gt = Image.fromarray(gt_np)

        # do cropping
        if self.crop_size is not None:
            crop_min_fg_ratio, crop_min_fg_prc = self.get_fg_sampling_params()

            valid_crop = False
            n_crop_tests = 0
            if ((crop_min_fg_ratio is None and crop_min_fg_prc is not None) or
                (crop_min_fg_ratio is not None and crop_min_fg_prc is None)):
                raise Exception("Inconsistent settings for Min FG Ratio and Min Ratio of FG focused patches")

            if crop_min_fg_ratio is None or crop_min_fg_prc <= 0.0:
                pick_hard_crop = False
            else:
                pick_hard_crop = self.augmentation_params.get_random() < crop_min_fg_ratio

            while not valid_crop:
                i, j, h, w = transforms.RandomCrop.get_params(pil_img, output_size=self.crop_size)
                tempo_crop_img = TF.crop(pil_img, i, j, h, w)
                if pil_gt is not None:
                    tempo_crop_gt = TF.crop(pil_gt, i, j, h, w)
                else:
                    # no ground truth for validation ... assume valid
                    tempo_crop_gt = None
                    valid_crop = True

                if crop_min_fg_prc is not None:
                    if pick_hard_crop:
                        # TODO: this might be affected by the padding bug described above ...
                        # try to pick a harder crop (foreground focus ...
                        crop_fg_percentage = (np.asarray(tempo_crop_gt) == 0).sum() / (self.crop_size[0] * self.crop_size[1])
                        # print(crop_fg_percentage)
                        valid_crop = crop_fg_percentage >= crop_min_fg_prc
                    else:
                        # randomly, accept this crop as is
                        valid_crop = True
                else:
                    # no validation, just stop here
                    valid_crop = True

                if n_crop_tests > 5:
                    # it has been tested more than 10 times without success...
                    #  keep it because the frame might not contain any valid crop
                    valid_crop = True

                if valid_crop:
                    # it is either valid or
                    pil_img = tempo_crop_img
                    pil_gt = tempo_crop_gt
                else:
                    n_crop_tests += 1

        if self.augmentation_params.should_apply_color_invert():
            # invert colors
            img_np = np.asarray(pil_img)
            img_np = 255 - img_np
            pil_img = Image.fromarray(img_np)

        if self.augmentation_params.should_apply_color_change():
            # transform color by using HUE
            pil_img = TF.adjust_hue(pil_img, (random.random() * 0.9 - 0.45))

        if self.augmentation_params.should_apply_gaussian_noise():
            # add gaussian noise
            img_np = np.asarray(pil_img).astype(np.float64)
            img_np += np.random.randn(img_np.shape[0], img_np.shape[1], img_np.shape[2]) * self.augmentation_params.gaussian_noise_range
            img_np[img_np < 0] = 0
            img_np[img_np > 255] = 255
            pil_img = Image.fromarray(img_np.astype(np.uint8))

        if self.augmentation_params.should_apply_luminosity_changes():
            # Apply random changes that affect the luminosity and Sharpness of the image
            aug_val = self.augmentation_params.get_TF_luminosity_value()
            pil_img = TF.adjust_brightness(pil_img, aug_val)

            aug_val = self.augmentation_params.get_TF_contrast_value()
            pil_img = TF.adjust_contrast(pil_img, aug_val)

            aug_val = self.augmentation_params.get_TF_gamma_value()
            pil_img = TF.adjust_gamma(pil_img, aug_val)

            aug_val = self.augmentation_params.get_TF_saturation_value()
            pil_img = TF.adjust_saturation(pil_img, aug_val)

        if self.text_region_mask_st is not None:
            # This is usually done for Stage 3.
            # The binarization GT is used to automatically generate a pseudo label for text regions
            # The resulting mask should be background=0 and foreground=255.
            mask_gt = np.asarray(pil_gt)
            mask_gt = 255 - cv2.erode(mask_gt, self.text_region_mask_st)
            mask_gt = Image.fromarray(mask_gt)

            text_mask_gt_t = TF.to_tensor(mask_gt)
        else:
            # Usually done in Stages 1 or 2, pil_gt represents the mask
            # it is still assume that background=0 and foreground=255
            mask_gt = None
            text_mask_gt_t = 0

        """
        # debugging
        debug_img = np.asarray(pil_img)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        debug_gt = np.asarray(pil_gt)
        cv2.imshow("Image", debug_img)
        cv2.imshow("GT", debug_gt)
        cv2.imshow("mask", np.asarray(mask_gt))
        cv2.waitKey()
        """

        # convert to tensor
        img_t = TF.to_tensor(pil_img)

        if self.reconstruct_type != "none":
            tempo_img = np.asarray(pil_img)
            if self.reconstruct_masked:
                pre_modified = tempo_img.copy()

            """
            copy_before = cv2.cvtColor(tempo_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("image before", copy_before)
            """

            if self.reconstruct_type == "median":
                tempo_img = cv2.medianBlur(tempo_img, self.reconstruct_filter_K)
            elif self.reconstruct_type == "mean":
                tempo_img = cv2.blur(tempo_img, (self.reconstruct_filter_K, self.reconstruct_filter_K))
            elif self.reconstruct_type == "gaussian":
                tempo_img = cv2.GaussianBlur(tempo_img, (self.reconstruct_filter_K, self.reconstruct_filter_K), 0)
            else:
                # assume raw ... nothing to do
                pass

            if self.reconstruct_masked:
                # TODO: should this be affected by text_region_mask_st?
                # TODO: yes, if eroded ground truth is being used, the mask here should work with the given one
                if mask_gt is None:
                    # no "masked" GT generated, assuming that GT is the mask
                    # that is usual behavior for text detector dataset
                    mask_gt = np.asarray(pil_gt)
                    overwrite_mask = True
                else:
                    # masked GT was prepared ... usually from pixel level GT
                    # that is usual behavior for binarization training
                    # use the prepared mask directly, and do NOT overwrite later
                    mask_gt = np.asarray(mask_gt)
                    overwrite_mask = False
                """
                cv2.imshow("mask", (mask_gt > 0).astype(np.uint8) * 255)
                """

                if self.reconstruct_type == "inpaint":
                    # pre_modified[mask_gt > 0] = 0
                    tempo_img = cv2.inpaint(pre_modified, mask_gt, self.reconstruct_filter_K, cv2.INPAINT_TELEA)

                    # use the difference between the original and inpaint to define a binarization target
                    # (e.g. the characters)
                    raw_diff = np.abs(pre_modified.astype(np.int32) - tempo_img.astype(np.int32)).astype(np.uint8)

                    # Older version, this one computed the difference using grayscale instead of max of 3-channel diffs
                    # gray_diff = cv2.cvtColor(raw_diff, cv2.COLOR_BGR2GRAY)
                    # _, bin_diff = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # bin_diff = 255 - bin_diff

                    # get the maximum difference (on any channel), and then binarize
                    max_diff = np.max(raw_diff, axis=2)
                    # this should be an approximation of the characters or anything that contrast with the background
                    _, bin_max_diff = cv2.threshold(max_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # for character GT, use the un-dilated version, inverted...
                    bin_diff = 255 - bin_max_diff
                    # now, dilate and make sure that all pixels are covered by the original text mask
                    dil_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (2, 2))
                    dilated_bin_max_diff = cv2.morphologyEx(bin_max_diff, cv2.MORPH_DILATE, dil_se)
                    refined_gt_mask = np.logical_and(dilated_bin_max_diff, mask_gt).astype(np.uint8) * 255

                    """
                    # this version uses the pixels believed to be characters, and inpaints those regions only
                    # However, this lead to noisier results, great in some cases, and terrible in many others 
                    # get a second round inpainted image ... that keeps background pixels closer to the original
                    inpaint_2 = cv2.inpaint(pre_modified, refined_gt_mask, self.reconstruct_filter_K, cv2.INPAINT_TELEA)
                    cv2.imshow("Inpaint 2", cv2.cvtColor(inpaint_2, cv2.COLOR_RGB2BGR))
                    inpaint_2_blurred = cv2.GaussianBlur(inpaint_2, (7, 7), 0)
                    inpaint_2[mask_gt > 0] = inpaint_2_blurred[mask_gt > 0]
                    cv2.imshow("Inpaint 2 (blurred)", cv2.cvtColor(inpaint_2, cv2.COLOR_RGB2BGR))
                    """

                    # create a refined version: use the refined diff to bring back background (replace inpainting)
                    # this works great in most cases, and terrible in others, where background is complex
                    # and where the there are background colors with higher contrast to the inpainted color than the
                    # text itself.
                    inpaint_2 = tempo_img.copy()
                    inpaint_2[refined_gt_mask == 0] = pre_modified[refined_gt_mask == 0]
                    # cv2.imshow("Inpaint 2", cv2.cvtColor(inpaint_2, cv2.COLOR_RGB2BGR))
                    # now, blur for better results ...
                    inpaint_2_blurred = cv2.GaussianBlur(inpaint_2, (15, 15), 0)
                    inpaint_2[mask_gt > 0] = inpaint_2_blurred[mask_gt > 0]
                    # cv2.imshow("Inpaint 2 (blurred)", cv2.cvtColor(inpaint_2, cv2.COLOR_RGB2BGR))

                    tempo_img = inpaint_2

                    if overwrite_mask:
                        # This over-writes the expanded mask (if any)
                        mask_gt = Image.fromarray(bin_diff)
                        text_mask_gt_t = TF.to_tensor(mask_gt)

                    """
                    cv2.imshow("characters raw", raw_diff)
                    cv2.imshow("bin characters", bin_diff)

                    cv2.imshow("max RGB diff with inpainted 1", max_diff)
                    cv2.imshow("Otsu of max RGB diff with inpainted 1", bin_max_diff)
                    cv2.imshow("Otsu of max RGB diff with inpainted 1 (dilated, refined)", refined_gt_mask)
                    # cv2.imshow("Inpaint 2", cv2.cvtColor(inpaint_2, cv2.COLOR_RGB2BGR))
                    """

                else:
                    # regular behavior, combine previous result with the mask
                    pre_modified[mask_gt > 0] = tempo_img[mask_gt > 0]
                    # and replace ...
                    tempo_img = pre_modified

            """
            cv2.imshow("image after", cv2.cvtColor(tempo_img, cv2.COLOR_RGB2BGR))            
            diff = ((copy_before.astype(np.int32) - cv2.cvtColor(tempo_img, cv2.COLOR_RGB2BGR).astype(np.int32)  + 128) / 2).astype(np.uint8)
            cv2.imshow("diff", diff)            
            cv2.waitKey()
            """

            if self.reconstruct_lap_pyramid:
                # decompose into per-frequency targets using a Laplacian Pyramid approach ...
                # compute the pyramid. Note that color normalizaiton is done there, after converting to flaot32
                laplacian_targets = compute_laplacian_targets(tempo_img, self.lap_pyramid_levels,
                                                              [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

                rec_target_t = {}
                for lap_level, lap_target in enumerate(laplacian_targets):
                    lap_transposed = lap_target.transpose((2, 0, 1)).copy()
                    lap_level_t = torch.from_numpy(lap_transposed).float()
                    rec_target_t[f"L{lap_level}"] = lap_level_t
            else:
                # return regular single image ...
                rec_target_pil = Image.fromarray(tempo_img)
                rec_target_t = TF.to_tensor(rec_target_pil)

                # color normalization
                # rec_target_t = TF.normalize(rec_target_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # simple output rage adjustment
                rec_target_t = TF.normalize(rec_target_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            # for type=="none"
            rec_target_t = 0

        if self.reconstruction_mode:
            # same normalization as input
            # ... RGB normalization ....
            # gt_t = TF.normalize(gt_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ... color adjustment ....
            gt_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            # convert the mask to tensor
            if pil_gt is not None:
                gt_t = TF.to_tensor(pil_gt)
            else:
                gt_t = 0

        # ... normalize the RGB values
        # img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ... no normalization ... simply re-scale to Tanh range ...
        img_t = TF.normalize(img_t, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # check if per pixel weights are required ....
        if self.weight_st_element is not None:
            # get weights ....
            weight_gt = np.asarray(pil_gt)
            eroded_gt = cv2.erode(weight_gt, self.weight_st_element)

            # start with uniform weights
            weights = np.ones(eroded_gt.shape, dtype=np.float64)
            # add extra proportion
            weights[eroded_gt == 0] += self.weight_fg_extra

            """
            print(total_fg_exp_pixels)
            print(fg_proportion)
            print(weights.min())
            print(weights.max())
            vis_weights = ((weights / weights.max()) * 255).astype(dtype=np.uint8)

            cv2.imshow("original", weight_gt)
            cv2.imshow("eroded", eroded_gt)
            cv2.imshow("weights", vis_weights)
            cv2.waitKey()
            """

            weights_t = torch.tensor(weights)
        else:
            # no weights
            weights_t = 0

        return img_t, gt_t, weights_t, text_mask_gt_t, rec_target_t
