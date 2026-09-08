
import time
import numpy as np
import cv2
from contextlib import nullcontext

from PIL import Image

import torch
import torch.nn as nn

# Shared with the original LectureNet
from ...lecturenet_v1.util import LectureNet_Util
from ...lecturenet_v1.FCN_lecturenet_dataset import LectureNet_DataSet

from LM_Tools.optimization.data_augmentation_parameters import DataAugmentationParameters
from LM_Tools.optimization.log import Log
from LM_Tools.optimization.optimization_manager import OptimizationManager
from LM_Tools.optimization.perceptual_loss import VGGPerceptualLoss, PixWeightedVGGPerceptualLoss


class Stage2TextDelTrainer:
    def __init__(self, config, segmentation_only):
        self.segmentation_only = segmentation_only
        self.DEVICE = config.get("LectureNet.General.Device", "cuda:0")

        self.clear_CUDA_CACHE = True

        self.txt_config = config.get_subconfig("LectureNet.Pretraining.TextDetection")
        rec_config = config.get_subconfig("LectureNet.Pretraining.Reconstruction")

        # Load datasets configuration ....
        crop_size_w = config.get("LectureNet.Training.CropSize.Width")
        crop_size_h = config.get("LectureNet.Training.CropSize.Height")
        self.kfbin_crop_size = (crop_size_h, crop_size_w)
        self.kfbin_weight_expansion = self.txt_config.get("Masks.WeightExpansion", 1)
        self.kfbin_weight_extra = self.txt_config.get("Masks.WeightForegroundExtra", 5.0)
        augmentation_config = self.txt_config.get_subconfig("Dataset.Augmentations")
        self.augmentation_params = DataAugmentationParameters.FromConfiguration(augmentation_config)
        self.rec_type = rec_config.get("Target.Type", "None")
        self.rec_filter_k = rec_config.get_int("Target.FilterK", 35)
        self.rec_masked = rec_config.get_bool("Target.Masked")

        self.train_images_dir = self.txt_config.get_str("Dataset.TrainImagesPath")
        self.train_masks_dir = self.txt_config.get_str("Dataset.TrainMasksPath")
        self.valid_images_dir = self.txt_config.get_str("Dataset.ValidImagesPath")
        self.valid_masks_dir = self.txt_config.get_str("Dataset.ValidMasksPath")
        self.pre_load_images = self.txt_config.get("Dataset.Preload", False)
        self.augment_validation = self.txt_config.get("Optimization.AugmentationsOnValidation", True)
        self.train_dataset = None
        self.valid_dataset = None

        # ... and the corresponding data loaders ...
        self.batch_size = self.txt_config.get("Optimization.BatchSize", 8)
        if self.txt_config.contains("Optimization.GradAccumulateStep"):
            # potentially using gradient accumulation ...
            self.grad_accumulate = self.txt_config.get("Optimization.GradAccumulateStep")
        else:
            # no gradient accumulation defined in the config .. simply assume none is needed
            self.grad_accumulate = 1

        self.loader_workers = self.txt_config.get("Optimization.LoaderWorkers", 0)
        self.persistent_workers = self.txt_config.get("Optimization.PersistentWorkers", False)
        self.train_loader = None
        self.valid_loader = None

        self.output_dir = config.get_str("LectureNet.General.OutputPath")
        self.pretrained_network_filename = self.txt_config.get_str("OutputFile", "FCN_PRETRAINED_TEXT.dat")

        self.debug_images_dir = self.txt_config.get_str("Dataset.DebugImagesPath", None)
        self.show_debug_images = self.txt_config.get_bool("Debug.SaveImages", False)
        self.debug_image_prefix = self.txt_config.get_str("Debug.SavePrefix", "DEBUG_REC_")
        self.full_debug_image_prefix = self.output_dir + "/" + self.debug_image_prefix
        self.debug_imgs = self.load_debug_images()

        # Log ...
        log_filename = self.txt_config.get_str("LogFile", "FCN_PRETRAINED_TEXT_log.txt")
        self.log = Log(self.output_dir + "/" + log_filename)
        self.log.to_log("Initializing Text Deletion Pretraining", add_time=True)

        if self.grad_accumulate > 1:
            self.log.to_log(
                f"-> Gradient Accumulation Enabled: Batches will be divided into {self.grad_accumulate}",
                add_time=True, display=True
            )

        # This one should be constructed later ... it needs a reference to the model
        self.op_manager_config = self.txt_config.get_subconfig("Optimization")
        self.op_manager = None

        # Loss functions ...
        # Main loss, used in all cases
        self.bce_mask_loss = nn.BCEWithLogitsLoss(reduction="mean")
        if self.segmentation_only:
            # when the model only does the main task, no auxiliary losses for other branches
            self.w_bin_loss = None
            self.w_tdet_loss = None
            self.w_tdel_loss = None
            self.rec_L1_w_text = None
            self.rec_L1_w_bg = None

            self.bce_binary_loss = None
            self.l1_loss_raw = None
            self.weighted_perceptual = None
            self.perception_loss = None
        else:
            self.w_bin_loss = self.txt_config.get("Optimization.Losses.Weights.Binarization")
            self.w_tdet_loss = self.txt_config.get("Optimization.Losses.Weights.TextDetection")
            self.w_tdel_loss = self.txt_config.get("Optimization.Losses.Weights.Background")
            self.rec_L1_w_text = self.txt_config.get("Optimization.Losses.Background.L1.WeightText")
            self.rec_L1_w_bg = self.txt_config.get("Optimization.Losses.Background.L1.WeightBackground")

            self.log.to_log(f"- Loss Weights: \n    Bin={self.w_bin_loss}", add_time=True, display=True)
            self.log.to_log(f"    Text Detect.: {self.w_tdet_loss}", add_time=True, display=True)
            self.log.to_log(f"    Text Deletion: {self.w_tdel_loss}", add_time=True, display=True)
            self.log.to_log(f"       W. Text: {self.rec_L1_w_text}", add_time=True, display=True)
            self.log.to_log(f"       W. Background: {self.rec_L1_w_bg}", add_time=True, display=True)

            self.bce_binary_loss = nn.BCEWithLogitsLoss(reduction="mean")
            # mse_loss = nn.MSELoss(reduction="mean")
            # mse_loss_raw = nn.MSELoss(reduction="none")
            self.l1_loss_raw = nn.L1Loss(reduction="none")

            self.weighted_perceptual = self.txt_config.get("Optimization.Losses.Background.Perceptual.Weighted")
            if self.weighted_perceptual:
                pos_text_weight = self.txt_config.get("Optimization.Losses.Background.Perceptual.WeightText")
                neg_text_weight = self.txt_config.get("Optimization.Losses.Background.Perceptual.WeightBackground")
                self.perception_loss = PixWeightedVGGPerceptualLoss(pos_text_weight, neg_text_weight).to(self.DEVICE)
                self.log.to_log(f"       Weighted Perceptual Loss", add_time=True, display=True)
                self.log.to_log(f"          W. Pos. Text: {pos_text_weight}", add_time=True, display=True)
                self.log.to_log(f"          W. Neg. Text: {neg_text_weight}", add_time=True, display=True)
            else:
                self.log.to_log(f"       Unweighted Perceptual Loss", add_time=True, display=True)
                self.perception_loss = VGGPerceptualLoss().to(self.DEVICE)

        # time
        self.elapsed_training = None
        self.use_AMP = self.txt_config.get("Optimization.UseAdaptiveMixedPrecision")

    def load_datasets(self):
        train_image_paths, train_masks_paths = LectureNet_Util.get_images_w_masks_filenames(
            self.train_images_dir, self.train_masks_dir, check_files=False
        )

        msg = "A total of {0:d} training images were found".format(len(train_image_paths))
        self.log.to_log(msg, display=True, add_time=True)

        self.train_dataset = LectureNet_DataSet(
            train_image_paths[:], train_masks_paths[:], False,
            crop_size=self.kfbin_crop_size, augmentation_params=self.augmentation_params,
            weight_expansion=self.kfbin_weight_expansion, weight_fg_extra=self.kfbin_weight_extra,
            text_region_masks_expansion=0, reconstruct_type=self.rec_type,
            reconstruct_filter_K=self.rec_filter_k, reconstruct_masked=self.rec_masked
        )

        valid_image_paths, valid_masks_paths = LectureNet_Util.get_images_w_masks_filenames(
            self.valid_images_dir, self.valid_masks_dir, check_files=False
        )

        msg = "A total of {0:d} validation images were found".format(len(valid_image_paths))
        self.log.to_log(msg, display=True, add_time=True)

        if self.augment_validation:
            valid_aug_params = self.augmentation_params
        else:
            valid_aug_params = None

        # creates a validation set with the same augmentations used for the training set
        self.valid_dataset = LectureNet_DataSet(
            valid_image_paths[:], valid_masks_paths[:], False,
            crop_size=self.kfbin_crop_size, augmentation_params=valid_aug_params,
            weight_expansion=self.kfbin_weight_expansion, weight_fg_extra=self.kfbin_weight_extra,
            text_region_masks_expansion=0, reconstruct_type=self.rec_type,
            reconstruct_filter_K=self.rec_filter_k, reconstruct_masked=self.rec_masked
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
        if not self.show_debug_images:
            # do not save ....
            return

        with torch.no_grad():
            lecture_net.eval()
            for idx, img in enumerate(self.debug_imgs):
                if self.segmentation_only:
                    reconstructed = lecture_net.reconstruct(img, False)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_MASK_{idx}_{epoch}.png", reconstructed)
                else:
                    binary, text_mask, rec_img = lecture_net.binarize(img, True)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_BIN_{idx}_{epoch}.png", binary)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_REC_{idx}_{epoch}.png", rec_img)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_MASK_{idx}_{epoch}.png", text_mask)

    def load_weights(self, lecture_net, has_skips):
        pretrained = self.txt_config.get("PretrainedModel.LoadWeights")

        if pretrained:
            pretrained_load_full = self.txt_config.get("PretrainedModel.LoadComplete")
            self.log.to_log(" -> Will train from pretrained model!", display=True, add_time=True)

            if pretrained_load_full:
                network_path = self.txt_config.get("PretrainedModel.Paths.Complete")
                self.log.to_log(f"Loading pretrained Full Network: {network_path}", display=True, add_time=True)
                lecture_net.load_state_dict(torch.load(network_path, map_location="cpu"))
            else:
                # load in parts ...
                encoder_path = self.txt_config.get("PretrainedModel.Paths.Encoder")
                self.log.to_log(f"Loading pretrained Encoder: {encoder_path}", display=True, add_time=True)
                lecture_net.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))

                if has_skips:
                    skips_path = self.txt_config.get("PretrainedModel.Paths.Skips")
                    self.log.to_log(f"Loading pretrained Skips: {skips_path}", display=True, add_time=True)
                    lecture_net.skips.load_state_dict(torch.load(skips_path, map_location="cpu"))

                decoder_path = self.txt_config.get("PretrainedModel.Paths.Decoder")
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

    def load_checkpoint(self, checkpoint_filename):
        msg = f"Loading checkpoint: {checkpoint_filename}"
        self.log.to_log(msg, display=True, add_time=True)
        self.op_manager.load_checkpoint(checkpoint_filename)

    def _multibranch_forward_and_loss(self, train_model, batch_data):
        # extract ....
        images, labels, weights, text_mask, medians = batch_data

        images = images.to(self.DEVICE)
        labels = labels.to(self.DEVICE)
        text_mask = text_mask.to(self.DEVICE)
        medians = medians.to(self.DEVICE)

        # NOTE: these no longer have sigmoid applied!!
        # ..... sigmoid will be required when USING regular BCE or MSE loss
        out_binary, out_text_mask, out_recons = train_model(images)

        # Train mask prediction using original ground truth (black = background, white=text)  ...
        mask_loss = self.bce_mask_loss(out_text_mask, labels) * self.w_tdet_loss

        # Train main branch
        binary_loss = self.bce_binary_loss(out_binary, text_mask) * self.w_bin_loss

        # Train reconstruction branch
        # Weighted loss, prioritizes the pixels within the text regions
        raw_rec_loss = self.l1_loss_raw(out_recons, medians)
        rec_loss = (raw_rec_loss * labels * self.rec_L1_w_text +
                    raw_rec_loss * (1 - labels) * self.rec_L1_w_bg).mean() * 1.0
        if self.weighted_perceptual:
            rec_loss = rec_loss + self.perception_loss(out_recons, medians, labels) * 1.0
        else:
            rec_loss = rec_loss + self.perception_loss(out_recons, medians) * 1.0

        loss = binary_loss + mask_loss + rec_loss * self.w_tdel_loss

        return loss, mask_loss, binary_loss, rec_loss

    def _singlebranch_forward_and_loss(self, train_model, batch_data):
        images, labels, weights, text_mask, medians = batch_data

        # get the inputs
        images = images.to(self.DEVICE)
        text_mask = text_mask.to(self.DEVICE)

        out_text_mask = train_model(images)

        # Train binarization using pseudo-labels ...
        bin_loss = self.bce_mask_loss(out_text_mask, text_mask)

        return bin_loss

    def train_loop(self, train_model):
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
                    loss, mask_loss, binary_loss, rec_loss = self._multibranch_forward_and_loss(train_model, batch_data)

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

    def valid_loop(self, train_model):
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
                    loss, mask_loss, binary_loss, rec_loss = self._multibranch_forward_and_loss(train_model, batch_data)

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
        start_training = time.time()
        for epoch in range(self.op_manager.current_epoch(), self.op_manager.max_epochs() + 1):
            self.log.to_log(f"Starting Epoch #{epoch}", display=True, add_time=True)

            epoch_train_loss = self.train_loop(train_model)
            epoch_valid_loss = self.valid_loop(train_model)

            msg = f"- E#{epoch}\tTrain. Loss: {epoch_train_loss:.8f}\t Val. Loss:{epoch_valid_loss:.8f}"
            self.log.to_log(msg, add_time=True)

            self.save_debug(epoch, train_model)

            # update the manager ...
            self.op_manager.epoch_update(epoch_valid_loss)
            if self.op_manager.last_improved():
                self.log.to_log(" -> Best model so far (overall validation loss reduced)",
                                display=False, add_time=True)

            if self.op_manager.early_stopping():
                # stop the training
                break
        end_training = time.time()
        self.elapsed_training = end_training - start_training

    def save_pretrained_model(self, lecture_net):
        lecture_net.eval()

        full_pretrained_network_filename = self.output_dir + "/" + self.pretrained_network_filename
        torch.save(lecture_net.state_dict(), full_pretrained_network_filename)

    def log_overall_times(self, elapsed_time):
        elapsed_loading = elapsed_time - self.elapsed_training
        self.log.to_log("Total time loading: " + str(elapsed_loading), display=True, add_time=True)
        self.log.to_log("Total time training: " + str(self.elapsed_training), display=True, add_time=True)
        self.log.to_log("Total time: " + str(elapsed_time), display=True, add_time=True)

        self.log.to_log("Text Detection and Deletion Pre-Training complete!", display=True, add_time=True)
