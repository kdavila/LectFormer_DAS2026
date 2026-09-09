
import os
import time
import math
import numpy as np

import cv2
from PIL import Image

import torch

class SegmentationEvaluator:
    def __init__(self, lecture_net, auto_encoder_mode, invert_output, tile_max_diagonal, tile_size, tile_overlap):
        self._lecture_net = lecture_net
        self._auto_encoder_mode = auto_encoder_mode
        self.invert_output = invert_output
        self._tile_max_diagonal = tile_max_diagonal
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        # initialize stats ...
        self._per_image_stats = {
            "peak_allocated_MB": [],
            "peak_reserved_MB": [],
            "current_allocated_MB": [],
            "current_reserved_MB": [],
            "heights": [],
            "widths": [],
            "time": []
        }

    @staticmethod
    def _compute_starts(length: int, tile_size: int, overlap: int):
        """
        Return tile start positions along one dimension, ensuring full coverage
        and that the last tile reaches the end.
        """
        if tile_size >= length:
            return [0]

        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError("overlap must be smaller than tile_size")

        starts = list(range(0, length - tile_size + 1, stride))

        last_start = length - tile_size
        if starts[-1] != last_start:
            starts.append(last_start)

        return starts

    @staticmethod
    def _make_1d_blend_weights(size: int, overlap: int, fade_left: bool, fade_right: bool):
        """
        Build 1D weights for one tile axis.

        - Outside overlap: weight = 1
        - In overlap with left neighbor: ramp 0 -> 1
        - In overlap with right neighbor: ramp 1 -> 0

        This makes adjacent tiles sum to 1 in overlap regions.
        """
        w = np.ones(size, dtype=np.float32)

        if overlap > 0:
            if fade_left:
                # Rising ramp on the left overlap
                w[:overlap] = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)

            if fade_right:
                # Falling ramp on the right overlap
                w[-overlap:] = np.minimum(
                    w[-overlap:],
                    np.linspace(1.0, 0.0, overlap, endpoint=True, dtype=np.float32)
                )

        return w

    @staticmethod
    def _make_tile_weight(h: int, w: int, overlap_y: int, overlap_x: int,
                          has_top: bool, has_bottom: bool, has_left: bool, has_right: bool):
        """
        2D separable weight map for a tile.
        Sum of overlapping neighboring tiles will be 1.
        """
        wy = SegmentationEvaluator._make_1d_blend_weights(
            size=h,
            overlap=min(overlap_y, h // 2 if h > 1 else 0),
            fade_left=has_top,
            fade_right=has_bottom
        )

        wx = SegmentationEvaluator._make_1d_blend_weights(
            size=w,
            overlap=min(overlap_x, w // 2 if w > 1 else 0),
            fade_left=has_left,
            fade_right=has_right
        )

        weight = wy[:, None] * wx[None, :]
        return weight.astype(np.float32)

    def _predict(self, in_img, binary_tresh=127):
        if self._auto_encoder_mode:
            return self._lecture_net.reconstruct(in_img, force_binary=True, binary_threshold=binary_tresh)
        else:
            return self._lecture_net.binarize(in_img, return_others=False, force_binary=True, binary_treshold=binary_tresh)

    def _reconstruct_tiled(self, img: Image.Image, tile_size: int = 1024, overlap: int = 256):
        """
        Tiled inference wrapper around:
            lecture_net.reconstruct(tile_img, force_binary=True, binary_threshold=127)

        Input:
            img: PIL.Image
        Output:
            numpy uint8 array, values in {0, 255}
        """
        W, H = img.size

        # Handle small images directly
        if W <= tile_size and H <= tile_size:
            return self._predict(img)

        x_starts = SegmentationEvaluator._compute_starts(W, tile_size, overlap)
        y_starts = SegmentationEvaluator._compute_starts(H, tile_size, overlap)

        acc = np.zeros((H, W), dtype=np.float32)
        acc_w = np.zeros((H, W), dtype=np.float32)

        for yi, y0 in enumerate(y_starts):
            for xi, x0 in enumerate(x_starts):
                x1 = min(x0 + tile_size, W)
                y1 = min(y0 + tile_size, H)

                tile = img.crop((x0, y0, x1, y1))

                pred = self._predict(tile)

                # Ensure float for blending
                pred = pred.astype(np.float32)

                h_tile, w_tile = pred.shape[:2]

                has_top = yi > 0
                has_bottom = yi < len(y_starts) - 1
                has_left = xi > 0
                has_right = xi < len(x_starts) - 1

                weight = SegmentationEvaluator._make_tile_weight(
                    h=h_tile, w=w_tile, overlap_y=overlap, overlap_x=overlap,
                    has_top=has_top, has_bottom=has_bottom, has_left=has_left, has_right=has_right
                )

                acc[y0:y1, x0:x1] += pred * weight
                acc_w[y0:y1, x0:x1] += weight

        # Safe normalization
        merged = acc / np.maximum(acc_w, 1e-8)

        # Since inputs were binary {0,255}, merged is in [0,255].
        # Re-binarize at midpoint.
        reconstructed = np.where(merged >= 127.5, 255, 0).astype(np.uint8)

        return reconstructed

    def reconstruct(self, img):
        # check if the image is too large
        diag = math.sqrt((img.width ** 2) + (img.height ** 2))
        if diag < self._tile_max_diagonal:
            return self._predict(img)
        else:
            print(f"-> Warning: the image is large (W={img.width} x H={img.height}), using tile-based prediction")

            # Method 0: run inference as usual
            # reconstructed = lecture_net.reconstruct(img, force_binary=True, binary_threshold=127)

            # Method 1: tile the reconstruction (this will bound the memory nicely)
            reconstructed = self._reconstruct_tiled(img, tile_size=self._tile_size, overlap=self._tile_overlap)

            # Method 2: simple resizing (simpler, but changes scales!, and does not impose a memory boundary)
            # img_small = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
            # rec_small = lecture_net.reconstruct(img_small, force_binary=True, binary_threshold=127)
            # reconstructed = cv2.resize(rec_small, (img.width, img.height), interpolation=cv2.INTER_NEAREST)

            return reconstructed

    @classmethod
    def clear_cache_and_stats(cls, device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    def capture_image_cuda_stats(self, img, img_time, device):
        torch.cuda.synchronize(device)

        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
        current_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
        current_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2

        self._per_image_stats["heights"].append(img.height)
        self._per_image_stats["widths"].append(img.width)
        self._per_image_stats["time"].append(img_time)
        self._per_image_stats["peak_allocated_MB"].append(peak_allocated)
        self._per_image_stats["peak_reserved_MB"].append(peak_reserved)
        self._per_image_stats["current_allocated_MB"].append(current_allocated)
        self._per_image_stats["current_reserved_MB"].append(current_reserved)

    def save_eval_images(self, all_paths, output_dir, device):
        self._lecture_net.eval()
        with torch.no_grad():
            for idx, path in enumerate(all_paths[:]):
                print(f"-> ({idx + 1} / {len(all_paths)}) Processing: {path}")

                base_dir, img_filename = os.path.split(path)
                base_img_name, ext = os.path.splitext(img_filename)
                output_path = f"{output_dir}/{base_img_name}.png"

                if os.path.exists(output_path):
                    print("-> WARNING: Image already exists! Skipping")
                    continue

                # loading ....
                img = Image.open(path)

                # prepare for capturing VRAM stats
                self.clear_cache_and_stats(device)

                start_img_time = time.time()

                reconstructed = self.reconstruct(img)

                end_img_time = time.time()
                img_time = end_img_time - start_img_time

                self.capture_image_cuda_stats(img, img_time, device)

                if self.invert_output:
                    reconstructed = 255 - reconstructed

                cv2.imwrite(output_path, reconstructed)

                """
                if idx % 10 == 0 and next(lecture_net.parameters()).is_cuda:
                    # clean the cache every so many images ....
                    torch.cuda.empty_cache()
                """

    def show_stats(self):
        print("stat:\tmin\tprc_25\tprc_50\tprc_75\tmax\tmean\tstdev")
        for stat in self._per_image_stats:
            raw_values = np.array(self._per_image_stats[stat])
            mean = raw_values.mean()
            stdev = raw_values.std()
            min = raw_values.min()
            max = raw_values.max()
            prc_25 = np.quantile(raw_values, 0.25)
            prc_50 = np.quantile(raw_values, 0.50)
            prc_75 = np.quantile(raw_values, 0.75)

            quart_str = f"{min:>8.2f} {prc_25:>8.2f} {prc_50:>8.2f} {prc_75:>8.2f} {max:>8.2f}"
            print(f"{stat}:\t{quart_str} {mean:>8.2f} {stdev:.2f}")

    @classmethod
    def textseg_fg_iou_fscore(cls, pred: np.ndarray, gt: np.ndarray, pred_text_value: int = 1, gt_text_value: int = 1,
                              ignore_label: int = 255):
        """
        Compute TextSeg-style foreground IoU and foreground F-score
        from pixel-level prediction and ground truth.

        Parameters
        ----------
        pred : Predicted label map, shape (H, W). Can be binary or label-valued.
        gt : Ground-truth label map, shape (H, W). Can be binary or label-valued.
        ignore_label : (int) GT pixels with this value are ignored.
        pred_text_value : (int)  Value in `pred` representing foreground text.
        gt_text_value : (int) Value in `gt` representing foreground text.

        Returns:
            dict with TP, FP, FN, fgIoU, fscore
        """
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

        valid = (gt != ignore_label)

        pred_fg = (pred == pred_text_value)
        gt_fg = (gt == gt_text_value)

        tp = np.logical_and(pred_fg, gt_fg) & valid
        fp = np.logical_and(pred_fg, ~gt_fg) & valid
        fn = np.logical_and(~pred_fg, gt_fg) & valid

        tp = int(tp.sum())
        fp = int(fp.sum())
        fn = int(fn.sum())

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn

        fg_iou = tp / denom_iou if denom_iou > 0 else 1.0
        fscore = 2 * tp / denom_f1 if denom_f1 > 0 else 1.0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fgIoU": fg_iou,
            "fscore": fscore,
            "recall": recall,
            "precision": precision
        }

    @staticmethod
    def CreateFromConfig(eval_config, lecture_net):
        use_auto_encoder = eval_config.get("AutoEncoderMode", False)

        invert_output = eval_config.get("InvertOutput", True)
        tile_max_diagonal = eval_config.get("Tiling.MaxDiagonal", 2400)
        tile_size = eval_config.get("Tiling.Size", 1024)
        tile_overlap = eval_config.get("Tiling.Overlap", 256)

        return SegmentationEvaluator(lecture_net, use_auto_encoder, invert_output,
                                     tile_max_diagonal, tile_size, tile_overlap)

