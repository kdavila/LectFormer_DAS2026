import os
import sys
sys.path.insert(0, '..')

import cv2
import numpy as np

from LectureMath.evaluation.segmentation_evaluator import SegmentationEvaluator

def save_visualization(in_img_filename, out_img_filename, gt_img, gt_text_value, pred_img, pred_text_value):
    img = cv2.imread(in_img_filename, cv2.IMREAD_GRAYSCALE)
    out_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    out_img[:, :, 0] = img
    out_img[:, :, 1] = img
    out_img[:, :, 2] = img
    out_img[pred_img == pred_text_value, 2] = 255
    out_img[gt_img == gt_text_value, 1] = 255
    cv2.imwrite(out_img_filename, out_img)

def main():
    if len(sys.argv) < 4:
        print("Usage")
        print(f"\tpython {sys.argv[0]:s} pred_dir gt_dir text_seg [img_dir] [out_dir]")
        print("With:")
        print("\tpred_dir\tPath to directory with predictions ")
        print("\tgt_dir\tPath to directory with ground truths")
        print("\ttext_seg\t1 if GT is in TextSeg format, 0 otherwise")
        print("\timg_dir\tOptional. Path to directory with original images")
        print("\tout_dir\tOptional. Path to directory where visualizations will be saved")
        return

    pred_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    try:
        text_seg_style = int(sys.argv[3]) > 0
    except:
        print("Invalid value for text_seg. Use 1 for true, 0 for false")
        return

    if len(sys.argv) >= 6:
        img_dir, out_dir = sys.argv[4], sys.argv[5]
    else:
        img_dir, out_dir = None, None

    # binarizers used here produce 0/255
    pred_text_value = 255
    if text_seg_style:
        # in TextSeg, the 255 represent "don't care"
        # and 100 represents text (200 is text effect)
        ignore_label, gt_text_value = 255, 100
    else:
        # by default, do not ignore anything
        # and assume image is also binary, with max value as foreground
        ignore_label, gt_text_value = -1, 255

    predicted_files = os.listdir(pred_dir)
    averages = {
        "fgIoU": 0.0,
        "fscore": 0.0,
        "recall": 0.0,
        "precision": 0.0
    }

    for idx, pred_filename in enumerate(predicted_files):
        print(f"({idx + 1} / {len(predicted_files)}) - Processing: {pred_filename}")
        base, ext = os.path.splitext(pred_filename)
        if text_seg_style:
            gt_filename = f"{base}_maskfg.png"
        else:
            gt_filename = f"{base}.png"

        pred_img = cv2.imread(f"{pred_dir}/{pred_filename}", cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(f"{gt_dir}/{gt_filename}", cv2.IMREAD_GRAYSCALE)

        if img_dir is not None:
            save_visualization(f"{img_dir}/{base}.jpg", f"{out_dir}/{pred_filename}",
                               gt_img, gt_text_value, pred_img, pred_text_value)

        img_scores = SegmentationEvaluator.textseg_fg_iou_fscore(
            pred_img, gt_img, pred_text_value, gt_text_value, ignore_label
        )
        print(img_scores)

        for metric in averages:
            averages[metric] += img_scores[metric]

    for metric in averages:
        averages[metric] /= len(predicted_files)
        print(f"Average {metric} = {averages[metric]:0.6f}")

if __name__ == "__main__":
    main()
