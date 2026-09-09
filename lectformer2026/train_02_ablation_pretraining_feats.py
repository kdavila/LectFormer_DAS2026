import sys
sys.path.insert(0, '..')

import time

import torch
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 32
else:
    torch._dynamo.config.cache_size_limit = 32

from LM_Tools.configuration.configuration import Configuration
from LM_Tools.optimization.fixed_schedule import FixedSchedule, ScheduleStage
from LectureMath.lecturenet_v2.model.lectformer import LectFormerAutoEncoder
from LectureMath.lecturenet_v2.training.stage3_segment_trainer import Stage3TextSegTrainer


def callback_optimize_params(model, learning_rate):
    print("-> Only the binarization head will be trained")
    return [
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate}
    ]


def callback_optimize_params_full(model, learning_rate):
    print("-> Full Network will be trained")
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate},
        {'params': model.skips.parameters(), 'lr': learning_rate},
        {'params': model.decoder.parameters(), 'lr': learning_rate},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate}
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage")
        print(f"\tpython {sys.argv[0]:s} config [chkpt_json_file]")
        print("With:")
        print("\tconfig\tPath to configuration file")
        return

    start_time = time.time()

    # read the config file
    config = Configuration.from_file(sys.argv[1], warning_mode=True)

    main_config_path = "LectureNet.Ablation.TextSegmentation"
    trainer = Stage3TextSegTrainer(config, main_config_path,
                                   training_name="Ablation - Segmentation (Auto-Encoder) with Fixed Schedule",
                                   lecture_mode=False, gt_invert=True, segmentation_only=True, heads_optional=False)

    trainer.load_datasets()
    trainer.create_data_loaders()

    # creating the model ....
    DEVICE = config.get("LectureNet.General.Device", "cuda:0")
    use_transformer_skips = not config.get("LectureNet.Network.Skips.BypassMode")
    use_middle_block = False
    print(f"\tUsing Middle Block: {use_middle_block}")

    lecture_net = LectFormerAutoEncoder.CreateFromConfig(
        config, 3, use_middle_block, use_transformer_skips,
        0.0, None, True
    )

    # Original definition of the Parts for this ablation script:
    # 1 - after stage 1, frozen backbone, only tweak binarization branch for 1000 epochs
    # 2 - after stage 2, zero shot (NOTHING TO DO HERE)
    # 3 - after stage 2, tweak everything for 500 epochs
    # 4 - after stage 1, tweak everything for 1000 epochs
    ablation_part = trainer.main_config.get("AblationParts.PretrainingFeatures")
    print(f"\t-Ablation Stage: {ablation_part}")

    if not trainer.main_config.get("PretrainedModel.LoadWeights"):
        print("\nWarning: Current configuration is inconsistent with the ablation study")
        print("         The configuration will not load a pretrained model, but a ")
        print("         Pretrained model is expected on each Part of the Ablation.\n")

    if ablation_part == 1:
        print("Part 1: After stage 1, frozen backbone, only train binarization branch for 1000 epochs")

        # long schedule
        schedule_stages = [
            ScheduleStage(0.001, 100),
            ScheduleStage(0.0001, 800),
            ScheduleStage(0.00001, 100),
        ]
        # only train binarization branch
        callback_func = callback_optimize_params

        # PART 1 only, freeze the backbone ...
        for p in lecture_net.encoder.parameters():
            p.requires_grad = False
        for p in lecture_net.skips.parameters():
            p.requires_grad = False
        for p in lecture_net.decoder.parameters():
            p.requires_grad = False
    elif ablation_part == 2:
        print("Part 2: Zero-shot performance after Stage 2. No fine-tuning needed")
        return
    elif ablation_part == 3:
        print("Part 3: After Stage 2, train everything for 500 epochs")

        # short schedule
        schedule_stages = [
            ScheduleStage(0.000100, 300),
            ScheduleStage(0.000010, 150),
            ScheduleStage(0.000001, 50),
        ]
        # and train everything
        callback_func = callback_optimize_params_full

        if not trainer.main_config.get("PretrainedModel.Paths.Complete"):
            print("Warning: A complete pretrained model produced by Stage 2 should be the input")
            print("         However, the pretrained model (if any) will be loaded in parts")

    elif ablation_part == 4:
        print("Part 4: After stage 1, train everything for 1000 epochs")

        # long schedule
        schedule_stages = [
            ScheduleStage(0.000100, 600),
            ScheduleStage(0.000010, 300),
            ScheduleStage(0.000001, 100),
        ]
        # and train everything
        callback_func = callback_optimize_params_full
    else:
        print(f"Error: Ablation Part {ablation_part} Undefined")
        return

    schedule = FixedSchedule(None, 0, None, schedule_stages)

    if (ablation_part == 1 or ablation_part == 4) and trainer.main_config.get("PretrainedModel.Paths.Complete"):
        print("Warning: A pretrained model produced by Stage 1 should be the input")
        print("         The model should be loaded in Parts. However, a complete pretrained model might be loaded")

    trainer.load_weights(lecture_net, use_transformer_skips)

    # move network to selected device
    lecture_net = lecture_net.to(DEVICE)
    trainer.log_trainable_parameters(lecture_net)

    # Get the optimization manager (optimizer, learning rate control, check points)
    trainer.create_op_manager(lecture_net, callback_func, None, schedule)

    if len(sys.argv) >= 3:
        trainer.load_checkpoint(sys.argv[2])

    use_compile = trainer.main_config.get("Optimization.CompileModel")
    if use_compile:
        # Caution: might be troublesome on Windows!
        trainer.perception_loss = torch.compile(trainer.perception_loss)

        # full model compilation = trouble!
        train_model = torch.compile(lecture_net)
    else:
        train_model = lecture_net

    # if debug images are enabled AND it's the first epoch (do not overwrite if training from checkpoint)
    if trainer.op_manager.current_epoch() == 1:
        trainer.save_debug(0, train_model)

    trainer.exec_training(train_model)

    trainer.save_trained_model(lecture_net)

    end_time = time.time()
    trainer.log_overall_times(end_time - start_time)


if __name__ == "__main__":
    main()
