import sys
sys.path.insert(0, "./..")

# from munkres import Munkres

import torch

from LM_Tools.configuration.configuration import Configuration

from LectureMath.lecturenet_v2.model.lectformer import LectFormer
from LectureMath.lecturenet_v1.util import LectureNet_Util
from LectureMath.evaluation.binary_IOU_det_evaluator import BinaryDetIoUEvaluator


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\tpython {0:s} config model".format(sys.argv[0]))
        print("Where")
        print("\tconfig\tPath to configuration file")
        print("\tmodel\tPath to network that will be evaluated")
        return

    config = Configuration.from_file(sys.argv[1], strict_mode=True)
    model_filename = sys.argv[2]

    DEVICE = config.get("LectureNet.General.Device", "cuda:0")

    eval_config = config.get_subconfig("LectureNet.Pretraining.TextDetection.Evaluation")

    images_dir = eval_config.get_str("ImagesPath")
    masks_dir = eval_config.get_str("MasksPath")

    all_image_filenames, all_mask_filenames = LectureNet_Util.get_images_w_masks_filenames(images_dir, masks_dir,
                                                                                           check_files=False)

    eval_IOU_t = eval_config.get("IOU.Thresholds", [0.5])
    bin_threshold = eval_config.get_int("BinarizationThreshold", 128)

    evaluator = BinaryDetIoUEvaluator(all_image_filenames, all_mask_filenames, eval_IOU_t)

    print("... loading model ...")
    lecture_net = LectFormer.CreateFromConfig(config, 3)
    lecture_net.load_state_dict(torch.load(model_filename))
    lecture_net.eval()

    lecture_net = lecture_net.to(DEVICE)

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print("Total Trainable Parameters in Network: " + str(pytorch_total_params))

    # create a function which captures the right closure (including the network)
    def binarize_image(in_image):
        with torch.no_grad():
            binary, text_mask, rec_img = lecture_net.binarize(in_image, return_others=True, force_binary=True,
                                                              binary_treshold=bin_threshold)
            # binary_image = 255 - binary_image
        return text_mask

    all_stats, all_pixel_stats, with_issues = evaluator.run_evaluation(binarize_image)

    if len(with_issues) > 0:
        print(f"\n\nImages with issues fixed: {len(with_issues):d}")
        print("List of images with issues")
        for img_name in with_issues:
            print(img_name)

    evaluator.print_eval_main_results(all_stats)
    evaluator.print_eval_pixel_results(all_pixel_stats)



if __name__ == "__main__":
    main()

