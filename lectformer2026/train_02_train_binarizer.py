
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
    print("-> Standard Learning Rate, Full Network In Training")
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate},
        {'params': model.skips.parameters(), 'lr': learning_rate},
        {'params': model.decoder.parameters(), 'lr': learning_rate},
        {'params': model.conv_text_mask_out.parameters(), 'lr': learning_rate},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate},
        {'params': model.conv_binarizer.parameters(), 'lr': learning_rate}
    ]


def callback_warmup_params(model, learning_rate):
    # Initially, only the output layer is updated ....
    print("-> Warmup Mode: Backbone is Frozen, only the branches are training")
    return [
        {'params': model.conv_text_mask_out.parameters(), 'lr': learning_rate},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate},
        {'params': model.conv_binarizer.parameters(), 'lr': learning_rate}
    ]


def main():
    if len(sys.argv) < 2:
        print("usage")
        print(f"\tpython {sys.argv[0]} config [checkpoint]")
        print("\n\nwhere")
        print("\tconfig:\tAccessMath Configuration File")
        return

    start_time = time.time()

    config = Configuration.from_file(sys.argv[1])

    main_config_path = "LectureNet.Training.Binarization"
    trainer = Stage3TextSegTrainer(config, main_config_path, training_name="Lecture Binarization", lecture_mode=True,
                                   gt_invert=False, segmentation_only=False, heads_optional=False)

    trainer.load_datasets()
    trainer.create_data_loaders()

    # creating the model ....
    DEVICE = config.get("LectureNet.General.Device", "cuda:0")
    lecture_net = LectFormer.CreateFromConfig(config, 3)
    trainer.load_weights(lecture_net, True)
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
