
import os
import time
import numpy as np
import cv2
from contextlib import nullcontext

from PIL import Image

import torch
import torch.nn as nn

from LectureMath.data.meta_data_DB import MetaDataDB

# Shared with the original LectureNet
from ...lecturenet_v1.util import LectureNet_Util
from ...lecturenet_v1.FCN_lecturenet_dataset import LectureNet_DataSet
from .fg_sampling_scheduler import ForegroundSamplingScheduler

from LM_Tools.optimization.data_augmentation_parameters import DataAugmentationParameters
from LM_Tools.optimization.log import Log
from LM_Tools.configuration.configuration import Configuration
from LM_Tools.optimization.optimization_manager import OptimizationManager
from LM_Tools.optimization.perceptual_loss import VGGPerceptualLoss, PixWeightedVGGPerceptualLoss


class Stage3TextSegTrainer:
    def __init__(self, config, main_config_path, training_name, lecture_mode, gt_invert, segmentation_only,
                 heads_optional):

        assert isinstance(config, Configuration)

        self.DEVICE = config.get("LectureNet.General.Device", "cuda:0")
        self.clear_CUDA_CACHE = True

        self.training_name = training_name
        self.lecture_mode = lecture_mode
        if self.lecture_mode:
            self.lecture_database_path = config.get_str("LectureNet.LectureDataset.DataBasePath")
        else:
            self.lecture_database_path = None

        self.gt_invert = gt_invert
        self.segmentation_only = segmentation_only
        self.heads_optional = heads_optional

        self.main_config = config.get_subconfig(main_config_path)

        # Load datasets configuration ....
        crop_size_w = config.get("LectureNet.Training.CropSize.Width")
        crop_size_h = config.get("LectureNet.Training.CropSize.Height")
        self.kfbin_crop_size = (crop_size_h, crop_size_w)
        self.kfbin_weight_expansion = self.main_config.get("Masks.WeightExpansion", 1)
        self.kfbin_weight_extra = self.main_config.get("Masks.WeightForegroundExtra", 5.0)
        if not self.segmentation_only:
            self.kfbin_text_masks_expansion = self.main_config.get_int("Masks.Expansion", 10)
        else:
            self.kfbin_text_masks_expansion = 0

        augmentation_config = self.main_config.get_subconfig("Dataset.Augmentations")
        self.augmentation_params = DataAugmentationParameters.FromConfiguration(augmentation_config)

        if not self.segmentation_only:
            rec_config = config.get_subconfig("LectureNet.Pretraining.Reconstruction")
            self.rec_type = rec_config.get("Target.Type", "None")
            self.rec_filter_k = rec_config.get_int("Target.FilterK", 35)
            self.rec_masked = rec_config.get_bool("Target.Masked")
        else:
            self.rec_type = None
            self.rec_filter_k = 0
            self.rec_masked = False

        self.pre_load_images = self.main_config.get("Dataset.Preload", False)
        self.augment_validation = self.main_config.get("Optimization.AugmentationsOnValidation", True)
        self.train_dataset = None
        self.valid_dataset = None

        # ... and the corresponding data loaders ...
        self.batch_size = self.main_config.get("Optimization.BatchSize", 8)
        if self.main_config.contains("Optimization.GradAccumulateStep"):
            # potentially using gradient accumulation ...
            self.grad_accumulate = self.main_config.get("Optimization.GradAccumulateStep")
        else:
            # no gradient accumulation defined in the config .. simply assume none is needed
            self.grad_accumulate = 1

        self.loader_workers = self.main_config.get("Optimization.LoaderWorkers", 0)
        self.persistent_workers = self.main_config.get("Optimization.PersistentWorkers", False)
        self.train_loader = None
        self.valid_loader = None

        self.output_dir = config.get_str("LectureNet.General.OutputPath")
        self.trained_network_filename = self.main_config.get_str("OutputFile", "FCN_TRAINED_TEXT_SEG.dat")

        self.debug_images_dir = self.main_config.get_str("Dataset.DebugImagesPath", None)
        self.show_debug_images = self.main_config.get_bool("Debug.SaveImages", False)
        if self.main_config.contains("Debug.Frequency"):
            self.debug_freq = self.main_config.get_int("Debug.Frequency")
        else:
            self.debug_freq = 1
        self.debug_image_prefix = self.main_config.get_str("Debug.SavePrefix", "DEBUG_REC_")
        self.full_debug_image_prefix = self.output_dir + "/" + self.debug_image_prefix
        self.debug_imgs = self.load_debug_images()

        # Log ...
        log_filename = self.main_config.get_str("LogFile", "FCN_PRETRAINED_TEXT_log.txt")
        self.log = Log(self.output_dir + "/" + log_filename)
        self.log.to_log("Initializing Training for: " + self.training_name, add_time=True, display=True)

        if self.grad_accumulate > 1:
            self.log.to_log(
                f"-> Gradient Accumulation Enabled: Batches will be divided into {self.grad_accumulate}",
                add_time=True, display=True
            )

        # This one should be constructed later ... it needs a reference to the model
        self.op_manager_config = self.main_config.get_subconfig("Optimization")
        self.op_manager = None

        # This one also needs to be constructed later, if required by the script
        self.fg_sample_scheduler = None

        # Loss functions ...
        # Main loss, used in all cases
        self.bce_binary_loss = nn.BCEWithLogitsLoss(reduction="mean")
        if self.segmentation_only:
            # when the model only does the main task, no auxiliary losses for other branches
            self.w_bin_loss = None
            self.w_tdet_loss = None
            self.w_tdel_loss = None
            self.rec_L1_w_text = None
            self.rec_L1_w_bg = None

            self.bce_mask_loss = None
            self.l1_loss_raw = None
            self.weighted_perceptual = None
            self.perception_loss = None
        else:
            self.w_bin_loss = self.main_config.get("Optimization.Losses.Weights.Binarization")
            self.w_tdet_loss = self.main_config.get("Optimization.Losses.Weights.TextDetection")
            self.w_tdel_loss = self.main_config.get("Optimization.Losses.Weights.Background")
            self.rec_L1_w_text = self.main_config.get("Optimization.Losses.Background.L1.WeightText")
            self.rec_L1_w_bg = self.main_config.get("Optimization.Losses.Background.L1.WeightBackground")

            self.log.to_log(f"- Loss Weights: \n    Bin={self.w_bin_loss}", add_time=True, display=True)
            self.log.to_log(f"    Text Detect.: {self.w_tdet_loss}", add_time=True, display=True)
            self.log.to_log(f"    Text Deletion: {self.w_tdel_loss}", add_time=True, display=True)
            self.log.to_log(f"       W. Text: {self.rec_L1_w_text}", add_time=True, display=True)
            self.log.to_log(f"       W. Background: {self.rec_L1_w_bg}", add_time=True, display=True)

            self.bce_mask_loss = nn.BCEWithLogitsLoss(reduction="mean")
            # mse_loss = nn.MSELoss(reduction="mean")
            # mse_loss_raw = nn.MSELoss(reduction="none")
            self.l1_loss_raw = nn.L1Loss(reduction="none")

            self.weighted_perceptual = self.main_config.get("Optimization.Losses.Background.Perceptual.Weighted")
            if self.weighted_perceptual:
                pos_text_weight = self.main_config.get("Optimization.Losses.Background.Perceptual.WeightText")
                neg_text_weight = self.main_config.get("Optimization.Losses.Background.Perceptual.WeightBackground")
                self.perception_loss = PixWeightedVGGPerceptualLoss(pos_text_weight, neg_text_weight).to(self.DEVICE)
                self.log.to_log(f"       Weighted Perceptual Loss", add_time=True, display=True)
                self.log.to_log(f"          W. Pos. Text: {pos_text_weight}", add_time=True, display=True)
                self.log.to_log(f"          W. Neg. Text: {neg_text_weight}", add_time=True, display=True)
            else:
                self.log.to_log(f"       Unweighted Perceptual Loss", add_time=True, display=True)
                self.perception_loss = VGGPerceptualLoss().to(self.DEVICE)

        # time
        self.elapsed_training = None
        self.use_AMP = self.main_config.get("Optimization.UseAdaptiveMixedPrecision")

    @classmethod
    def _extract_lecture_annotation_pairs(cls, root_dir, database, dataset_name):
        training_set = database.get_dataset(dataset_name)

        all_images_locations = []
        all_gt_locations = []
        for lecture in training_set:
            # print(lecture.title.lower())
            annotation_prefix = root_dir + "/" + database.output_annotations + "/" + database.name + "_" + lecture.title.lower()

            annot_image_dir = annotation_prefix + "/keyframes"
            annot_binary_dir = annotation_prefix + "/binary"

            lecture_image_elements = os.listdir(annot_image_dir)
            lecture_binary_elements = os.listdir(annot_binary_dir)

            for img_filename in lecture_image_elements:
                base, ext = os.path.splitext(img_filename)

                if ext.lower() == ".png":
                    # keyframe image ... find in binary gt
                    if img_filename in lecture_binary_elements:
                        # valid key-frame (has Ground Truth) ... add!
                        all_images_locations.append(annot_image_dir + "/" + img_filename)
                        all_gt_locations.append(annot_binary_dir + "/" + img_filename)

        return all_images_locations, all_gt_locations

    def _load_lecture_datasets_paths(self):
        # load the database
        try:
            database = MetaDataDB.from_file(self.lecture_database_path)
        except Exception as e:
            print("Invalid database file")
            print(e)
            return

        # this will only work if the main configuration is for Binarization
        train_dataset_name = self.main_config.get("Dataset.TrainDatasetName")
        valid_dataset_name = self.main_config.get("Dataset.ValidDatasetName")

        train_image_paths, train_masks_paths = self._extract_lecture_annotation_pairs(self.output_dir, database, train_dataset_name)
        valid_image_paths, valid_masks_paths = self._extract_lecture_annotation_pairs(self.output_dir, database, valid_dataset_name)

        return (train_image_paths, train_masks_paths), (valid_image_paths, valid_masks_paths)

    def _load_segmentation_datasets_paths(self):
        train_images_dir = self.main_config.get_str("Dataset.TrainImagesPath")
        train_masks_dir = self.main_config.get_str("Dataset.TrainMasksPath")
        valid_images_dir = self.main_config.get_str("Dataset.ValidImagesPath")
        valid_masks_dir = self.main_config.get_str("Dataset.ValidMasksPath")

        train_image_paths, train_masks_paths = LectureNet_Util.get_images_w_masks_filenames(
            train_images_dir, train_masks_dir, check_files=False
        )

        valid_image_paths, valid_masks_paths = LectureNet_Util.get_images_w_masks_filenames(
            valid_images_dir, valid_masks_dir, check_files=False
        )

        return (train_image_paths, train_masks_paths), (valid_image_paths, valid_masks_paths)

    def load_datasets(self):
        if self.lecture_mode:
            train_paths, valid_paths = self._load_lecture_datasets_paths()
        else:
            train_paths, valid_paths = self._load_segmentation_datasets_paths()

        train_image_paths, train_masks_paths = train_paths
        valid_image_paths, valid_masks_paths = valid_paths

        msg = "A total of {0:d} training images were found".format(len(train_image_paths))
        self.log.to_log(msg, display=True, add_time=True)

        msg = "A total of {0:d} validation images were found".format(len(valid_image_paths))
        self.log.to_log(msg, display=True, add_time=True)

        # print("\n\n\nIMAGES ARE BEING LIMITED!!!\n\n\n")
        # MAX_IMGS = 300

        self.train_dataset = LectureNet_DataSet(
            train_image_paths[:], train_masks_paths[:], False, crop_size=self.kfbin_crop_size,
            crop_remove_empty_borders=False, crop_min_fg_prc=None, augmentation_params=self.augmentation_params,
            weight_expansion=self.kfbin_weight_expansion, weight_fg_extra=self.kfbin_weight_extra,
            text_region_masks_expansion=self.kfbin_text_masks_expansion, reconstruct_type=self.rec_type,
            reconstruct_filter_K=self.rec_filter_k, reconstruct_masked=self.rec_masked, invert_gt=self.gt_invert
        )

        if self.augment_validation:
            valid_aug_params = self.augmentation_params
        else:
            valid_aug_params = None

        # creates a validation set with the same augmentations used for the training set
        self.valid_dataset = LectureNet_DataSet(
            valid_image_paths[:], valid_masks_paths[:], False, crop_size=self.kfbin_crop_size,
            crop_remove_empty_borders=False, crop_min_fg_prc=None, augmentation_params=valid_aug_params,
            weight_expansion=self.kfbin_weight_expansion, weight_fg_extra=self.kfbin_weight_extra,
            text_region_masks_expansion=self.kfbin_text_masks_expansion, reconstruct_type=self.rec_type,
            reconstruct_filter_K=self.rec_filter_k, reconstruct_masked=self.rec_masked, invert_gt=self.gt_invert
        )

        if self.pre_load_images:
            print("Pre-loading training images")
            self.train_dataset.preload()

            print("Pre-loading validation images")
            self.valid_dataset.preload()
        else:
            print("Images will not be pre-loaded!")

    def create_data_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size // self.grad_accumulate, shuffle=True,
            num_workers=self.loader_workers, persistent_workers=self.persistent_workers
        )

        # TODO: can validation always handle larger batch when grad_accumlate > 1?
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.loader_workers, persistent_workers=self.persistent_workers
        )

    def load_debug_images(self):
        # for DEBUGGING
        if self.show_debug_images and self.debug_images_dir is not None:
            tempo_paths = LectureNet_Util.get_only_images_filenames(self.debug_images_dir)
            debug_imgs = [Image.open(tempo_path) for tempo_path in tempo_paths]
        else:
            debug_imgs = []

        return debug_imgs

    def save_debug(self, epoch, lecture_net):
        if not self.show_debug_images or (epoch % self.debug_freq != 0):
            # do not save ....
            return

        with torch.no_grad():
            lecture_net.eval()
            for idx, img in enumerate(self.debug_imgs):
                if self.segmentation_only:
                    reconstructed = lecture_net.reconstruct(img, False)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_MASK_{idx}_{epoch}.png", reconstructed)
                else:
                    binary, text_mask, rec_img = lecture_net.binarize(img, return_others=True, force_binary=False)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_BIN_{idx}_{epoch}.png", binary)
                    if rec_img is not None:
                        cv2.imwrite(f"{self.full_debug_image_prefix}_REC_{idx}_{epoch}.png", rec_img)
                    if text_mask is not None:
                        cv2.imwrite(f"{self.full_debug_image_prefix}_MASK_{idx}_{epoch}.png", text_mask)

    def load_weights(self, lecture_net, has_skips):
        pretrained = self.main_config.get("PretrainedModel.LoadWeights")

        if pretrained:
            pretrained_load_full = self.main_config.get("PretrainedModel.LoadComplete")
            self.log.to_log(" -> Will train from pretrained model!", display=True, add_time=True)

            if pretrained_load_full:
                network_path = self.main_config.get("PretrainedModel.Paths.Complete")
                self.log.to_log(f"Loading pretrained Full Network: {network_path}", display=True, add_time=True)
                lecture_net.load_state_dict(torch.load(network_path, map_location="cpu"))
            else:
                # load in parts ...
                encoder_path = self.main_config.get("PretrainedModel.Paths.Encoder")
                self.log.to_log(f"Loading pretrained Encoder: {encoder_path}", display=True, add_time=True)
                lecture_net.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))

                if has_skips:
                    skips_path = self.main_config.get("PretrainedModel.Paths.Skips")
                    self.log.to_log(f"Loading pretrained Skips: {skips_path}", display=True, add_time=True)
                    lecture_net.skips.load_state_dict(torch.load(skips_path, map_location="cpu"))

                decoder_path = self.main_config.get("PretrainedModel.Paths.Decoder")
                self.log.to_log(f"Loading pretrained Decoder: {decoder_path}", display=True, add_time=True)
                lecture_net.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        else:
            self.log.to_log(" -> Will train model from Scratch!", display=True, add_time=True)

    def log_trainable_parameters(self, lecture_net):
        pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
        msg = "Total Trainable Parameters in Network: " + str(pytorch_total_params)
        self.log.to_log(msg, display=True, add_time=True)

    def create_op_manager(self, lecture_net, callback_main, callback_warmup, schedule):
        self.op_manager = OptimizationManager.FromConfiguration(
            lecture_net, self.op_manager_config, self.output_dir, True, callback_main, callback_warmup,
            fixed_schedule=schedule
        )

    def create_fg_sampling_scheduler(self):
        self.fg_sample_scheduler = ForegroundSamplingScheduler.CreateFromConfig(self.main_config, self.op_manager)
        self.fg_sample_scheduler.log_settings(self.log)

    def load_checkpoint(self, checkpoint_filename):
        if self.fg_sample_scheduler is None:
            load_diff_state = None
        else:
            load_diff_state = {"CurrentMinForeground": None, "Active": None}

        msg = f"Loading checkpoint: {checkpoint_filename}"
        self.log.to_log(msg, display=True, add_time=True)
        self.op_manager.load_checkpoint(checkpoint_filename, checkpoint_load=load_diff_state)

        if self.fg_sample_scheduler is not None:
            # restore the difficulty
            active = load_diff_state["Active"]
            saved_difficulty = load_diff_state["CurrentMinForeground"]
            self.fg_sample_scheduler.set_from_checkpoint(active, saved_difficulty, self.train_dataset, self.log)

    def _multibranch_forward_and_loss(self, train_model, batch_data, bg_head_disabled, txt_head_disabled):
        # extract ....
        images, labels, weights, text_mask, textless_bg = batch_data

        # get the inputs
        images = images.to(self.DEVICE)            # raw input
        labels = labels.to(self.DEVICE)            # binary GT (black fg, white bg)
        # weights = weights.to(self.DEVICE)
        text_mask = text_mask.to(self.DEVICE)      # dilated bin GT (a text mask, black bg, white fg)
        textless_bg = textless_bg.to(self.DEVICE)  # estimated bg (No Text)

        # NOTE: Main branch and Text Branch do not include a sigmoid layer at the end in order to be compatible
        #       with BCEWithLogits loss, but sigmoid will be required when USING regular BCE or MSE loss
        out_binary, out_text_mask, out_recons = train_model(images)

        # (BCE)
        # binary_loss = bce_loss(out_binary, labels)
        binary_loss = self.bce_binary_loss(out_binary, labels) * self.w_bin_loss

        if not txt_head_disabled:
            # compute txt loss as usual ....
            mask_loss = self.bce_mask_loss(out_text_mask, text_mask) * self.w_tdet_loss
        else:
            # bypass text mask loss
            mask_loss = torch.tensor([0.0], dtype=binary_loss.dtype, device=binary_loss.device)

        if not bg_head_disabled:
            raw_rec_loss = self.l1_loss_raw(out_recons, textless_bg)
            rec_loss = (raw_rec_loss * text_mask * self.rec_L1_w_text +
                        raw_rec_loss * (1 - text_mask) * self.rec_L1_w_bg).mean() * 1.0
            if self.weighted_perceptual:
                rec_loss = rec_loss + self.perception_loss(out_recons, textless_bg, text_mask) * 1.0
            else:
                rec_loss = rec_loss + self.perception_loss(out_recons, textless_bg) * 1.0
        else:
            # bypass bg loss
            rec_loss = torch.tensor([0.0], dtype=binary_loss.dtype, device=binary_loss.device)

        loss = binary_loss + mask_loss + rec_loss * self.w_tdel_loss

        return loss, mask_loss, binary_loss, rec_loss

    def _singlebranch_forward_and_loss(self, train_model, batch_data):
        images, labels, weights, text_mask, textless_bg = batch_data

        # get the inputs
        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)

        out_text_mask = train_model(images)

        # Train mask prediction using original ground truth (black = background, white=text)  ...
        bin_loss = self.bce_binary_loss(out_text_mask, labels)

        return bin_loss

    def train_loop(self, train_model, bg_head_disabled, txt_head_disabled):
        epoch_train_loss = 0.0
        epoch_train_bin_loss = 0.0
        epoch_train_rec_loss = 0.0
        epoch_train_txt_loss = 0.0
        train_model.train()

        for i, batch_data in enumerate(self.train_loader, 0):
            print(f"T: {i + 1}/{len(self.train_loader)}", end="\r")

            if self.use_AMP:
                context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                context = nullcontext()

            with context:
                if self.segmentation_only:
                    loss = self._singlebranch_forward_and_loss(train_model, batch_data)
                    binary_loss = loss
                else:
                    loss, mask_loss, binary_loss, rec_loss = self._multibranch_forward_and_loss(
                        train_model, batch_data, bg_head_disabled, txt_head_disabled
                    )

                if self.grad_accumulate > 1:
                    # split by updates
                    loss = loss / self.grad_accumulate
                else:
                    # normal loss scale
                    pass

            loss.backward()

            if self.grad_accumulate == 1 or ((i + 1) % self.grad_accumulate == 0):
                # clipping the norm for large gradients
                # nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)

                self.op_manager.optimizer().step()
                self.op_manager.optimizer().zero_grad()

            # print statistics
            epoch_train_loss += loss.item()
            epoch_train_bin_loss += binary_loss.item()
            if not self.segmentation_only:
                epoch_train_rec_loss += rec_loss.item()
                epoch_train_txt_loss += mask_loss.item()

            print(f"T: {i + 1}/{len(self.train_loader)} - Loss={loss.item()}", end="\r")

        if self.DEVICE[:4] == "cuda" and self.clear_CUDA_CACHE:
            torch.cuda.empty_cache()

        epoch_train_loss /= (len(self.train_dataset))
        epoch_train_bin_loss /= (len(self.train_dataset))
        print(" - Epoch Training Loss: " + str(epoch_train_loss))
        if self.segmentation_only:
            msg = f" - Train Losses: Bin={epoch_train_bin_loss}"
            self.log.to_log(msg, display=True, add_time=True)
        else:
            epoch_train_rec_loss /= (len(self.train_dataset))
            epoch_train_txt_loss /= (len(self.train_dataset))

            msg = f" - Train Losses: Bin={epoch_train_bin_loss}, Rec={epoch_train_rec_loss}, Text={epoch_train_txt_loss}"
            self.log.to_log(msg, display=True, add_time=True)

        return epoch_train_loss

    def valid_loop(self, train_model, bg_head_disabled, txt_head_disabled):
        epoch_valid_loss = 0.0
        epoch_valid_bin_loss = 0.0
        epoch_valid_rec_loss = 0.0
        epoch_valid_txt_loss = 0.0
        train_model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(self.valid_loader, 0):

                if self.segmentation_only:
                    loss = self._singlebranch_forward_and_loss(train_model, batch_data)
                    binary_loss = loss
                else:
                    loss, mask_loss, binary_loss, rec_loss = self._multibranch_forward_and_loss(
                        train_model, batch_data, bg_head_disabled, txt_head_disabled
                    )

                epoch_valid_loss += loss.item()
                epoch_valid_bin_loss += binary_loss.item()
                if not self.segmentation_only:
                    epoch_valid_rec_loss += rec_loss.item()
                    epoch_valid_txt_loss += mask_loss.item()

                print(f"V: {i + 1}/{len(self.valid_loader)} - Loss={loss.item()}", end="\r")

        if self.DEVICE[:4] == "cuda" and self.clear_CUDA_CACHE:
            torch.cuda.empty_cache()

        epoch_valid_loss /= (len(self.valid_dataset))
        epoch_valid_bin_loss /= (len(self.valid_dataset))
        print(" - Epoch Validation Loss: " + str(epoch_valid_loss))
        if self.segmentation_only:
            msg = f" - Valid Losses: Bin={epoch_valid_bin_loss}"
            self.log.to_log(msg, display=True, add_time=True)
        else:
            epoch_valid_rec_loss /= (len(self.valid_dataset))
            epoch_valid_txt_loss /= (len(self.valid_dataset))

            msg = f" - Train Losses: Bin={epoch_valid_bin_loss}, Rec={epoch_valid_rec_loss}, Text={epoch_valid_txt_loss}"
            self.log.to_log(msg, display=True, add_time=True)

        return epoch_valid_loss

    def exec_training(self, train_model):
        if self.segmentation_only:
            # No auxiliary heads available during training
            bg_head_disabled = True
            txt_head_disabled = True
        else:
            if self.heads_optional:
                # Depending on the configuration, the auxiliary heads might be absent
                bg_head_disabled = train_model.is_bg_head_disabled()
                txt_head_disabled = train_model.is_txt_head_disabled()
            else:
                # assume that auxiliary heads should be available
                bg_head_disabled = False
                txt_head_disabled = False

        start_training = time.time()
        for epoch in range(self.op_manager.current_epoch(), self.op_manager.max_epochs() + 1):
            self.log.to_log(f"Starting Epoch #{epoch}", display=True, add_time=True)

            epoch_train_loss = self.train_loop(train_model, bg_head_disabled, txt_head_disabled)
            epoch_valid_loss = self.valid_loop(train_model, bg_head_disabled, txt_head_disabled)

            msg = f"- E#{epoch}\tTrain. Loss: {epoch_train_loss:.8f}\t Val. Loss:{epoch_valid_loss:.8f}"
            self.log.to_log(msg, add_time=True)

            self.save_debug(epoch, train_model)

            if self.fg_sample_scheduler is not None:
                # potentially increase the difficulty....
                self.fg_sample_scheduler.update_difficulty(self.train_dataset, self.log)
                save_diff_state = {
                    "Active": self.fg_sample_scheduler.active,
                    "CurrentMinForeground": self.fg_sample_scheduler.current_min_fg_prc
                }
            else:
                save_diff_state = None

            # update the manager ...
            self.op_manager.epoch_update(epoch_valid_loss, save_diff_state)
            if self.op_manager.last_improved():
                self.log.to_log(" -> Best model so far (overall validation loss reduced)",
                                display=False, add_time=True)

            if self.op_manager.early_stopping():
                # stop the training
                break
        end_training = time.time()
        self.elapsed_training = end_training - start_training

    def save_trained_model(self, lecture_net):
        lecture_net.eval()

        full_trained_network_filename = self.output_dir + "/" + self.trained_network_filename
        torch.save(lecture_net.state_dict(), full_trained_network_filename)

    def log_overall_times(self, elapsed_time):
        elapsed_loading = elapsed_time - self.elapsed_training
        self.log.to_log("Total time loading: " + str(elapsed_loading), display=True, add_time=True)
        self.log.to_log("Total time training: " + str(self.elapsed_training), display=True, add_time=True)
        self.log.to_log("Total time: " + str(elapsed_time), display=True, add_time=True)

        self.log.to_log(f"Training: {self.training_name} complete!", display=True, add_time=True)
