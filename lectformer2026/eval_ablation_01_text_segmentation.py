import sys
sys.path.insert(0, '..')

import time

import torch

from LM_Tools.configuration.configuration import Configuration

from LectureMath.lecturenet_v2.model.lectformer import LectFormerAutoEncoder, LectFormer
from LectureMath.lecturenet_v1.util import LectureNet_Util
from LectureMath.evaluation.segmentation_evaluator import SegmentationEvaluator


def main():
    if len(sys.argv) < 5:
        print("Usage")
        print(f"\tpython {sys.argv[0]:s} config weights input_dir output_dir")
        print("With:")
        print("\tconfig\tPath to configuration file")
        print("\tweights\tPath to trained model weights")
        print("\tinput_dir\tPath to directory with input images")
        print("\toutput_dir\tPath to directory for output binary images")
        return

    start_time = time.time()
    start_loading = time.time()

    config_filename = sys.argv[1]
    weights_filename = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]

    # read the config file
    config = Configuration.from_file(config_filename, warning_mode=True, strict_mode=True)

    DEVICE = config.get("LectureNet.General.Device", "cuda:0")

    # NOTE THAT THIS WORKS DIFFERENT FOR AUTO-ENCODER AND FOR FULL NETWORK!
    use_transformer_skips = not config.get("LectureNet.Network.Skips.BypassMode")
    eval_config = config.get_subconfig("LectureNet.Ablation.TextSegmentation.Evaluation")

    use_auto_encoder = eval_config.get("AutoEncoderMode", False)
    use_middle_block = eval_config.get("AutoEncoderMiddleBlock", False)

    if use_auto_encoder:
        lecture_net = LectFormerAutoEncoder.CreateFromConfig(
            config, 3, use_middle_block, use_transformer_skips,
            0.0, None, True
        )
    else:
        lecture_net = LectFormer.CreateFromConfig(config, 3)

    pytorch_total_params = sum(p.numel() for p in lecture_net.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters in Network: {pytorch_total_params}")

    # load trained network
    print(f"Loading Network: {weights_filename}")
    lecture_net.load_state_dict(torch.load(weights_filename, map_location="cpu"))

    evaluator = SegmentationEvaluator.CreateFromConfig(eval_config, lecture_net)
    print(f"\tInverting Output: {evaluator.invert_output}")

    end_loading = time.time()
    start_evaluation = time.time()

    # move network to selected device
    lecture_net = lecture_net.to(DEVICE)

    all_in_paths = LectureNet_Util.get_only_images_filenames(input_dir)
    print(f"- A total of {len(all_in_paths)} images were found!")

    evaluator.save_eval_images(all_in_paths, output_dir, DEVICE)

    end_evaluation = time.time()
    end_time = time.time()

    evaluator.show_stats()

    print(f"Total time loading: {end_loading - start_loading}")
    print(f"Total time training: {end_evaluation - start_evaluation}")
    print(f"Total time: {end_time - start_time}")
    print("Text Detection Ablation Binarization Complete!")


if __name__ == "__main__":
    main()
