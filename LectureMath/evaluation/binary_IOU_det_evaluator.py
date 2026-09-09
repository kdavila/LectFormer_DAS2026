
from PIL import Image, ImageOps

import cv2
import numpy as np


class BinaryDetIoUEvaluator:
    def __init__(self, all_image_filenames, all_mask_filenames, eval_IOU_t):
        # all files and their corresponding ground truths ...
        self.image_filenames = all_image_filenames
        self.mask_filenames = all_mask_filenames

        # list of IoU thresholds to consider ....
        self.eval_IOU_t = eval_IOU_t

    @classmethod
    def compute_match_candidates(cls, gt_labels, gt_stats, gt_n_labels, out_labels, out_stats, out_n_labels,
                                 min_IOU_threshold, min_cc_size):
        all_pairwise_valid_matches = []
        for out_idx in range(1, out_n_labels):
            if out_stats[out_idx, 4] >= min_cc_size:
                # the CC is large enough to be a valid match ....
                # but is possible that it does not overlap any CC ... check
                out_cc_x, out_cc_y, out_cc_w, out_cc_h, out_cc_size = out_stats[out_idx]
                out_cc_mask = (out_labels == out_idx)

                # ... against every CC in GT
                for gt_idx in range(1, gt_n_labels):
                    gt_cc_x, gt_cc_y, gt_cc_w, gt_cc_h, gt_cc_size = gt_stats[gt_idx]

                    # check for overlap between CC (COARSE)
                    if ((out_cc_x < gt_cc_x + gt_cc_w and gt_cc_x < out_cc_x + out_cc_w) and
                            (out_cc_y < gt_cc_y + gt_cc_h and gt_cc_y < out_cc_y + out_cc_h)):

                        # quantify pixel-level overlap between CC (FINE)
                        gt_cc_mask = (gt_labels == gt_idx)

                        intersection = np.logical_and(out_cc_mask, gt_cc_mask)
                        union = np.logical_or(out_cc_mask, gt_cc_mask)

                        intersection_size = intersection.sum()
                        union_size = union.sum()

                        # GET IOU
                        IOU = intersection_size / union_size
                        # print((out_idx, gt_idx, IOU))

                        # ONLY consider for matching if IOU is large enough anyway
                        if IOU >= min_IOU_threshold:
                            all_pairwise_valid_matches.append((IOU, gt_idx, out_idx))

        return all_pairwise_valid_matches

    def initialize_visualizations(self, gt_binary, out_binary):
        # initialize the dictionary of visualizations
        visualization_images = {}
        for iou_t in self.eval_IOU_t:
            visualization_images[iou_t] = np.zeros((gt_binary.shape[0], gt_binary.shape[1], 3), np.uint8)
            visualization_images[iou_t][:, :, 0] = gt_binary.copy()
            visualization_images[iou_t][:, :, 2] = out_binary.copy()

        return visualization_images

    def compute_assignments(self, all_pairwise_valid_matches, gt_labels, visualization_images):
        # Initialize the dictionary of valid matches
        valid_matches_per_threshold = {}
        for iou_t in self.eval_IOU_t:
            valid_matches_per_threshold[iou_t] = {"matches": 0}

        # ... sort match candidates by decreasing threshold
        all_pairwise_valid_matches = sorted(all_pairwise_valid_matches, reverse=True)
        matched_gt = {}
        matched_out = {}
        for IOU, gt_idx, out_idx in all_pairwise_valid_matches:
            # check if match between two elements not matched before
            if (gt_idx not in matched_gt) and (out_idx not in matched_out):
                # mark both of them as matched
                matched_gt[gt_idx] = True
                matched_out[out_idx] = True

                # only count the match for the cases where IOU surpasses the corresponding threshold
                # if it gets here, it should be at least as large as the first Threshold
                for iou_t in self.eval_IOU_t:
                    if IOU >= iou_t:
                        valid_matches_per_threshold[iou_t]["matches"] += 1

                        if visualization_images is not None:
                            gt_cc_mask = (gt_labels == gt_idx)
                            visualization_images[iou_t][gt_cc_mask, 1] = 255

        return valid_matches_per_threshold

    def compute_valid_matches_metrics(self, gt_n_labels, out_n_labels, matches_per_threshold):
        for iou_t in self.eval_IOU_t:
            if gt_n_labels > 1:
                recall = matches_per_threshold[iou_t]["matches"] / (gt_n_labels - 1)
            else:
                recall = 1.0

            if out_n_labels > 1:
                precision = matches_per_threshold[iou_t]["matches"] / (out_n_labels - 1)
            else:
                if gt_n_labels > 1:
                    precision = 0.0
                else:
                    precision = 1.0

            if recall + precision > 0.0:
                f1 = (2 * recall * precision) / (recall + precision)
            else:
                f1 = 0.0

            matches_per_threshold[iou_t]["recall"] = recall
            matches_per_threshold[iou_t]["precision"] = precision
            matches_per_threshold[iou_t]["f1"] = f1

    @classmethod
    def compute_per_pixel_stats(cls, gt_binary, out_binary):
        pixel_matches = np.logical_and(out_binary, gt_binary).sum()
        gt_fg_pixels = gt_binary.sum() / 255
        out_fg_pixels = out_binary.sum() / 255

        pixel_stats = {}
        if gt_fg_pixels > 0:
            pixel_stats["recall"] = pixel_matches / gt_fg_pixels
        else:
            pixel_stats["recall"] = 1.0

        if out_fg_pixels > 0:
            pixel_stats["precision"] = pixel_matches / out_fg_pixels
        else:
            if gt_fg_pixels > 0:
                pixel_stats["precision"] = 0.0
            else:
                pixel_stats["precision"] = 1.0

        if pixel_stats["recall"] + pixel_stats["precision"] > 0.0:
            pixel_stats["f1"] = (2 * pixel_stats["recall"] * pixel_stats["precision"]) / (
                    pixel_stats["recall"] + pixel_stats["precision"])
        else:
            pixel_stats["f1"] = 0.0

        return pixel_stats

    def compute_matching(self, out_binary, gt_binary, get_visualization=False):
        # 1) label CCs and get their boundaries on binary image
        # N includes background CC (0) ...
        # stats are N x 5 = (x, y, w, h, size)
        # centroids are N x 2 = (x, y)
        # ... for the output to evaluate ..,
        out_data = cv2.connectedComponentsWithStats(out_binary, connectivity=4)
        out_n_labels, out_labels, out_stats, out_centroids = out_data
        # ... for the ground truth used for evaluation ..,
        gt_data = cv2.connectedComponentsWithStats(gt_binary, connectivity=4)
        gt_n_labels, gt_labels, gt_stats, gt_centroids = gt_data

        # 2) find the size of smallest GT CC and the size of smallest prediction to match with current min IOU_threshold
        min_gt_size = gt_stats[:, 4].min()
        min_IOU = min(self.eval_IOU_t)
        min_cc_size = min_IOU * min_gt_size

        # 3) find matches...
        all_pairwise_valid_matches = self.compute_match_candidates(gt_labels, gt_stats, gt_n_labels, out_labels,
                                                                   out_stats, out_n_labels, min_IOU, min_cc_size)

        # 4) Optional, initialize visualizations
        if get_visualization:
            visualization_images = self.initialize_visualizations(gt_binary, out_binary)
        else:
            visualization_images = None

        # 5) count valid assignments which have IOU over thresholds
        matches_per_threshold = self.compute_assignments(all_pairwise_valid_matches, gt_labels, visualization_images)

        # 6) compute precision, recall and F-measure for all IOU thresholds
        self.compute_valid_matches_metrics(gt_n_labels, out_n_labels, matches_per_threshold)

        # 7) Finally, compute the pixel-level metrics ...
        pixel_stats = self.compute_per_pixel_stats(gt_binary, out_binary)

        if get_visualization:
            return matches_per_threshold, pixel_stats, visualization_images
        else:
            return matches_per_threshold, pixel_stats

    @classmethod
    def load_image(cls, img_filename):
        pil_image = Image.open(img_filename)
        o_w, o_h = pil_image.size
        try:
            pil_image = ImageOps.exif_transpose(pil_image)
        except:
            # mark as image that could not be properly loaded, that was changed and still has issues
            return None, True

        changed = False

        n_w, n_h = pil_image.size
        if o_w != n_w:
            changed = True

        if pil_image.mode == "CMYK" or pil_image.mode == "L":
            pil_image = pil_image.convert('RGB')
            changed = True

        return pil_image, changed

    @classmethod
    def accumulate_metrics(cls, src_metrics_dict, dst_metrics_dict):
        dst_metrics_dict["recall"].append(src_metrics_dict["recall"])
        dst_metrics_dict["precision"].append(src_metrics_dict["precision"])
        dst_metrics_dict["f1"].append(src_metrics_dict["f1"])

    def run_evaluation(self, bin_funct):
        with_issues = []
        all_stats = {iou_t: {"recall": [], "precision": [], "f1": []} for iou_t in self.eval_IOU_t}
        all_pixel_stats = {"recall": [], "precision": [], "f1": []}
        for idx, (img_filename, mask_filename) in enumerate(zip(self.image_filenames[:], self.mask_filenames[:])):
            print("Processing: " + img_filename + " (" + mask_filename + ")", flush=True)

            # load images
            pil_image, changed = self.load_image(img_filename)
            if changed:
                # the file had to be modified ....
                with_issues.append(img_filename)
            if pil_image is None:
                # and the file could not be loaded, do not process it further
                continue

            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

            # print(mask.shape)

            # print("... binarizing ... ", end="")
            text_mask = bin_funct(pil_image)

            # print("... matching ... ", end="")
            image_matches, pixel_stats = self.compute_matching(text_mask, mask)
            """
            image_matches, pixel_stats, visuals = evaluator.compute_matching(text_mask, mask, True)
            for vis_IOU in visuals:
                cv2.imshow(f"IoU={vis_IOU}", visuals[vis_IOU])
            cv2.waitKey()
            """

            for IOU_t in self.eval_IOU_t:
                self.accumulate_metrics(image_matches[IOU_t], all_stats[IOU_t])

            self.accumulate_metrics(pixel_stats, all_pixel_stats)

            # torch.cuda.empty_cache()

        return all_stats, all_pixel_stats, with_issues

    def print_eval_main_results(self, all_stats):
        print("\n\nEvaluation Metrics")
        print("IOU_t\tRec\tPrec\tF1")
        for IOU_t in self.eval_IOU_t:
            avg_recall = np.mean(all_stats[IOU_t]["recall"]) * 100.0
            avg_precision = np.mean(all_stats[IOU_t]["precision"]) * 100.0
            avg_f1 = np.mean(all_stats[IOU_t]["f1"]) * 100.0

            print(f"{IOU_t:.2f}\t{avg_recall:.2f}\t{avg_precision:.2f}\t{avg_f1:.2f}")

    def print_eval_pixel_results(self, all_pixel_stats):
        print(f"\n\nPixel Recall: {np.mean(all_pixel_stats['recall']) * 100.0:.2f}")
        print(f"Pixel Precision: {np.mean(all_pixel_stats['precision']) * 100.0:.2f}")
        print(f"Pixel F1: {np.mean(all_pixel_stats['f1']) * 100.0:.2f}")

