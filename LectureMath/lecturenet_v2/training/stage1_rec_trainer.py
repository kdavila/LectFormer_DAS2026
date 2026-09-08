
import time
import numpy as np
import cv2

from PIL import Image

import torch
import torch.nn as nn

# Shared with the original LectureNet
from ...lecturenet_v1.util import LectureNet_Util
from ...lecturenet_v1.FCN_lecturenet_dataset import LectureNet_DataSet

from LM_Tools.optimization.data_augmentation_parameters import DataAugmentationParameters
from LM_Tools.optimization.log import Log
from LM_Tools.optimization.optimization_manager import OptimizationManager
from LM_Tools.optimization.perceptual_loss import VGGPerceptualLoss


class Stage1RecTrainer:
    def __init__(self, config):
        self.DEVICE = config.get("LectureNet.General.Device", "cuda:0")
        self.clear_CUDA_CACHE = True

        rec_config = config.get_subconfig("LectureNet.Pretraining.Reconstruction")

        # Load datasets configuration ....
        crop_size_w = config.get("LectureNet.Training.CropSize.Width")
        crop_size_h = config.get("LectureNet.Training.CropSize.Height")
        self.kfbin_crop_size = (crop_size_h, crop_size_w)
        augmentation_config = rec_config.get_subconfig("Dataset.Augmentations")
        self.augmentation_params = DataAugmentationParameters.FromConfiguration(augmentation_config)
        self.rec_type = rec_config.get("AutoEncoder.TargetType")
        self.use_laplacian_pyr = rec_config.get("AutoEncoder.Laplacian.Enabled")
        self.laplacian_levels = rec_config.get("AutoEncoder.Laplacian.Levels")
        self.rec_filter_k = rec_config.get_int("Target.FilterK", 35)

        self.train_images_dir = rec_config.get_str("Dataset.TrainImagesPath")
        self.valid_images_dir = rec_config.get_str("Dataset.ValidImagesPath")
        self.pre_load_images = rec_config.get("Dataset.Preload", False)
        self.augment_validation = rec_config.get("Optimization.AugmentationsOnValidation", True)
        self.train_dataset = None
        self.valid_dataset = None

        # ... and the corresponding data loaders ...
        self.batch_size = rec_config.get("Optimization.BatchSize", 8)
        if rec_config.contains("Optimization.GradAccumulateStep"):
            # potentially using gradient accumulation ...
            self.grad_accumulate = rec_config.get("Optimization.GradAccumulateStep")
        else:
            # no gradient accumulation defined in the config .. simply assume none is needed
            self.grad_accumulate = 1
        self.loader_workers = rec_config.get("Optimization.LoaderWorkers", 0)
        self.persistent_workers = rec_config.get("Optimization.PersistentWorkers", False)
        self.train_loader = None
        self.valid_loader = None

        self.output_dir = config.get_str("LectureNet.General.OutputPath")
        self.pretrained_network_filename = rec_config.get_str("OutputFile", "FCN_PRETRAINED_REC.dat")

        # debugging images ...
        self.debug_images_dir = rec_config.get_str("Dataset.DebugImagesPath", None)
        self.show_debug_images = rec_config.get_bool("Debug.SaveImages", False)
        self.debug_image_prefix = rec_config.get_str("Debug.SavePrefix", "DEBUG_REC_")
        self.full_debug_image_prefix = self.output_dir + "/" + self.debug_image_prefix
        self.debug_imgs = self.load_debug_images()

        # Log ...
        log_filename = rec_config.get_str("LogFile", "FCN_PRETRAINED_REC_log.txt")
        self.log = Log(self.output_dir + "/" + log_filename)
        self.log.to_log("Initializing Reconstruction Pretraining", add_time=True)

        if self.grad_accumulate > 1:
            self.log.to_log(
                f"-> Gradient Accumulation Enabled: Batches will be divided into {self.grad_accumulate}",
                add_time=True, display=True
            )

        # This one should be constructed later ... it needs a reference to the model
        self.op_manager_config = rec_config.get_subconfig("Optimization")
        self.op_manager = None

        # Loss functions ...
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.perception_loss = VGGPerceptualLoss().to(self.DEVICE)

        # time
        self.elapsed_training = None

    def load_datasets(self):
        train_image_paths = LectureNet_Util.get_only_images_filenames(self.train_images_dir)

        msg = "A total of {0:d} training images were found".format(len(train_image_paths))
        self.log.to_log(msg, display=True, add_time=True)

        self.train_dataset = LectureNet_DataSet(
            train_image_paths[:], None, True, crop_size=self.kfbin_crop_size,
            augmentation_params=self.augmentation_params,
            reconstruct_type=self.rec_type, reconstruct_filter_K=self.rec_filter_k, reconstruct_masked=False,
            reconstruct_lap_pyramid=self.use_laplacian_pyr, lap_pyramid_levels=self.laplacian_levels
        )

        valid_image_paths = LectureNet_Util.get_only_images_filenames(self.valid_images_dir)

        msg = "A total of {0:d} validation images were found".format(len(valid_image_paths))
        self.log.to_log(msg, display=True, add_time=True)

        if self.augment_validation:
            valid_aug_params = self.augmentation_params
        else:
            valid_aug_params = None

        # creates a validation set with the same augmentations used for the training set
        self.valid_dataset = LectureNet_DataSet(
            valid_image_paths[:], None, True, crop_size=self.kfbin_crop_size,
            augmentation_params=valid_aug_params,
            reconstruct_type=self.rec_type, reconstruct_filter_K=self.rec_filter_k, reconstruct_masked=False
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
                reconstructed = lecture_net.reconstruct(img)
                cv2.imwrite(f"{self.full_debug_image_prefix}_{idx}_{epoch}.png", reconstructed)
                reconstructed = reconstructed.astype(np.float64)
                # this is needed because the internal mask was padded, but the output removes the padding!
                # so, remove the padding from the mask before using it
                if lecture_net.last_mask is not None:
                    tempo_mask = lecture_net.last_mask[:reconstructed.shape[0], :reconstructed.shape[1]]
                    reconstructed[tempo_mask > 0.5] *= 0.2
                    reconstructed = reconstructed.astype(np.uint8)
                    cv2.imwrite(f"{self.full_debug_image_prefix}_{idx}_masked_{epoch}.png", reconstructed)

    def load_and_freeze_encoder(self, lecture_net, encoder_path):
        # load pretrained encoder
        print(f"\nLoading PRETRAINED encoder from: {encoder_path}")
        lecture_net.encoder.load_state_dict(torch.load(encoder_path))
        print("\n\n- PRETRAINED encoder is FROZEN\n")
        # freezing the encoder
        lecture_net.encoder.eval()
        for param in lecture_net.encoder.parameters():
            param.requires_grad = False

    def log_trainable_parameters(self, lecture_net):
        pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
        msg = "Total Trainable Parameters in Network: " + str(pytorch_total_params)
        self.log.to_log(msg, display=True, add_time=True)

    def create_op_manager(self, lecture_net, callback_main, callback_warmup):
        self.op_manager = OptimizationManager.FromConfiguration(
            lecture_net, self.op_manager_config, self.output_dir, True, callback_main, callback_warmup
        )

    def load_checkpoint(self, checkpoint_filename):
        msg = f"Loading checkpoint: {checkpoint_filename}"
        self.log.to_log(msg, display=True, add_time=True)
        self.op_manager.load_checkpoint(checkpoint_filename)

    def _prepare_laplacian_labels(self, per_level_targets):
        per_level_labels = {}
        labels = None
        for level_key in per_level_targets:
            per_level_labels[level_key] = per_level_targets[level_key].to(self.DEVICE)
            if labels is None:
                labels = per_level_labels[level_key]
            else:
                labels += per_level_labels[level_key]

        return labels, per_level_labels

    def train_loop(self, lecture_net):
        epoch_train_loss = 0.0
        lecture_net.train()
        for i, (images, labels, weights, text_mask, medians) in enumerate(self.train_loader, 0):
            # print("mini batch {0:d}".format(i), flush=True)
            # get the inputs
            images = images.to(self.DEVICE)
            if self.rec_type.lower() != "None":
                if self.use_laplacian_pyr:
                    labels, per_level_labels = self._prepare_laplacian_labels(medians)
                else:
                    # use medians as the target
                    labels = medians.to(self.DEVICE)
            else:
                # use original image as target
                labels = labels.to(self.DEVICE)

            out_reconstruction = lecture_net(images)

            # Train reconstruction branch
            # Labels should contain same image using normalization with mean=(0.5,0.5,0.5) and std=(0.5,0.5,0.5)
            # so all values should be between [-1, 1] as produced by Tanh activations
            if self.use_laplacian_pyr:
                # per-level reconstruction
                rec_loss = 0.0
                final_rec = None
                for level_idx in range(self.laplacian_levels):
                    level_map_s = level_idx * 3
                    level_map_e = level_map_s + 3
                    level_rec = out_reconstruction[:, level_map_s:level_map_e, :, :]
                    rec_loss += self.l1_loss(level_rec, per_level_labels[f"L{level_idx}"])

                    if level_idx == 0:
                        final_rec = level_rec
                    else:
                        # add the maps to get the final reconstruction
                        final_rec = final_rec + level_rec

                # general reconstruction ...
                rec_loss += self.perception_loss(final_rec, labels) + self.l1_loss(final_rec, labels)
            else:
                # TODO: make this a switch from config (L1 or MSE)
                # rec_loss = mse_loss(out_reconstruction, labels)
                rec_loss = self.perception_loss(out_reconstruction, labels) + self.l1_loss(out_reconstruction, labels)

            # loss = binary_loss + mask_loss
            if self.grad_accumulate > 1:
                # split by updates
                loss = rec_loss / self.grad_accumulate
            else:
                # normal loss scale
                loss = rec_loss

            loss.backward()

            if self.grad_accumulate == 1 or ((i + 1) % self.grad_accumulate == 0):
                # clipping the norm for large gradients
                # nn.utils.clip_grad_norm_(lecture_net.parameters(), 1.0)

                self.op_manager.optimizer().step()
                self.op_manager.optimizer().zero_grad()

            # print statistics
            epoch_train_loss += loss.item()

            print(f"T: {i + 1}/{len(self.train_loader)} - Loss={loss.item()}", end="\r")

        if self.DEVICE[:4] == "cuda" and self.clear_CUDA_CACHE:
            torch.cuda.empty_cache()

        # epoch_loss /= (len(lecture_kf_dataset) * kfbin_crop_size[0] * kfbin_crop_size[1])
        epoch_train_loss /= (len(self.train_dataset))
        print(" - Epoch Training Loss: " + str(epoch_train_loss))

        return epoch_train_loss

    def valid_loop(self, lecture_net):
        epoch_valid_loss = 0.0
        lecture_net.eval()
        with torch.no_grad():
            for i, (images, labels, weights, text_mask, medians) in enumerate(self.valid_loader, 0):
                print(f"V: {i + 1}/{len(self.valid_loader)}", end="\r")

                # print("mini batch {0:d}".format(i), flush=True)
                # get the inputs
                images = images.to(self.DEVICE)
                if self.rec_type.lower() != "None":
                    # use medians as the target
                    labels = medians.to(self.DEVICE)
                else:
                    # use original image as target
                    labels = labels.to(self.DEVICE)

                out_reconstruction = lecture_net(images)

                if self.use_laplacian_pyr:
                    # per-level reconstruction
                    final_rec = None
                    for level_idx in range(self.laplacian_levels):
                        level_map_s = level_idx * 3
                        level_map_e = level_map_s + 3
                        level_rec = out_reconstruction[:, level_map_s:level_map_e, :, :]

                        if level_idx == 0:
                            final_rec = level_rec
                        else:
                            # add the maps to get the final reconstruction
                            final_rec = final_rec + level_rec
                    # replace by the final reconstruction
                    out_reconstruction = final_rec

                # TODO: make this a switch from config
                # rec_loss = mse_loss(out_reconstruction, labels)
                rec_loss = self.perception_loss(out_reconstruction, labels) + self.l1_loss(out_reconstruction, labels)

                # loss = binary_loss + mask_loss
                loss = rec_loss

                # print statistics
                epoch_valid_loss += loss.item()

        if self.DEVICE[:4] == "cuda" and self.clear_CUDA_CACHE:
            torch.cuda.empty_cache()

        epoch_valid_loss /= (len(self.valid_dataset))
        print(" - Epoch Validation Loss: " + str(epoch_valid_loss))

        return epoch_valid_loss

    def exec_training(self, lecture_net):
        start_training = time.time()
        for epoch in range(self.op_manager.current_epoch(), self.op_manager.max_epochs() + 1):
            self.log.to_log(f"Starting Epoch #{epoch}", display=True, add_time=True)

            epoch_train_loss = self.train_loop(lecture_net)
            epoch_valid_loss = self.valid_loop(lecture_net)

            msg = f"- E#{epoch}\tTrain. Loss: {epoch_train_loss:.8f}\t Val. Loss:{epoch_valid_loss:.8f}"
            self.log.to_log(msg, add_time=True)

            self.save_debug(epoch, lecture_net)

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

    def save_pretrained_model(self, lecture_net, use_transformer_skips):
        lecture_net.eval()

        full_pretrained_network_filename = self.output_dir + "/" + self.pretrained_network_filename
        full_pretrained_encoder_filename = full_pretrained_network_filename + ".encoder.dat"
        full_pretrained_decoder_filename = full_pretrained_network_filename + ".decoder.dat"

        torch.save(lecture_net.state_dict(), full_pretrained_network_filename)
        torch.save(lecture_net.encoder.state_dict(), full_pretrained_encoder_filename)
        torch.save(lecture_net.decoder.state_dict(), full_pretrained_decoder_filename)

        if use_transformer_skips:
            full_pretrained_skips_filename = full_pretrained_network_filename + ".skips.dat"
            torch.save(lecture_net.skips.state_dict(), full_pretrained_skips_filename)

    def log_overall_times(self, elapsed_time):
        elapsed_loading = elapsed_time - self.elapsed_training
        self.log.to_log("Total time loading: " + str(elapsed_loading), display=True, add_time=True)
        self.log.to_log("Total time training: " + str(self.elapsed_training), display=True, add_time=True)
        self.log.to_log("Total time: " + str(elapsed_time), display=True, add_time=True)

        self.log.to_log("Reconstruction Pre-Training complete!", display=True, add_time=True)
