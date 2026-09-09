import sys
sys.path.insert(0, '..')

import time

import torch
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 32
else:
    torch._dynamo.config.cache_size_limit = 32

from LM_Tools.configuration.configuration import Configuration
from LectureMath.lecturenet_v2.model.lectformer import LectFormer
from LectureMath.lecturenet_v2.training.stage3_segment_trainer import Stage3TextSegTrainer


def callback_optimize_params(model, learning_rate):
    # this function is used to set custom learning rates, usually smaller learning rates on encoder
    assert isinstance(model, LectFormer)

    # backbone + binarization head are always fixed ....
    param_set = [
        {'params': model.encoder.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.skips.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.decoder.parameters(), 'lr': learning_rate},
        {'params': model.conv_binarizer.parameters(), 'lr': learning_rate}
    ]
    # heads are ... optional
    if not model.is_bg_head_disabled():
        param_set.append({'params': model.conv_reconstruct.parameters(), 'lr': learning_rate})
    if not model.is_txt_head_disabled():
        param_set.append({'params': model.conv_text_mask_out.parameters(), 'lr': learning_rate})

    print("-> Standard Learning Rate. Full network (minus disabled heads) in Training")
    return param_set


def callback_warmup_params(model, learning_rate):
    assert isinstance(model, LectFormer)

    # NO backbone, but binarization is always fixed ....
    param_set = [
        {'params': model.conv_binarizer.parameters(), 'lr': learning_rate}
    ]
    # heads are ... optional
    if not model.is_bg_head_disabled():
        param_set.append({'params': model.conv_reconstruct.parameters(), 'lr': learning_rate})
    if not model.is_txt_head_disabled():
        param_set.append({'params': model.conv_text_mask_out.parameters(), 'lr': learning_rate})

    # Initially, only the output layer is updated ....
    print("\n- Warmup Mode: Backbone is Frozen, only active heads will be trained!\n")
    return param_set


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
    trainer = Stage3TextSegTrainer(config, main_config_path, training_name="Ablation - Heads on Text Segmentation",
                                   lecture_mode=False, gt_invert=True, segmentation_only=False, heads_optional=True)

    trainer.load_datasets()
    trainer.create_data_loaders()

    # creating the model ....
    DEVICE = config.get("LectureNet.General.Device", "cuda:0")
    use_transformer_skips = not config.get("LectureNet.Network.Skips.BypassMode")
    lecture_net = LectFormer.CreateFromConfig(config, 3)
    trainer.load_weights(lecture_net, use_transformer_skips)
    # move network to selected device
    lecture_net = lecture_net.to(DEVICE)
    trainer.log_trainable_parameters(lecture_net)

    # Get the optimization manager (optimizer, learning rate control, check points)
    trainer.create_op_manager(lecture_net, callback_optimize_params, callback_warmup_params, None)

    trainer.create_fg_sampling_scheduler()

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
