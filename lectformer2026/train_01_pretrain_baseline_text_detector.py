
import sys
sys.path.insert(0, '..')

import copy
import time

import torch
torch._dynamo.config.recompile_limit = 32

from LM_Tools.configuration.configuration import Configuration
from LectureMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
from LectureMath.lecturenet_v2.training.stage2_textdel_trainer import Stage2TextDelTrainer


def callback_optimize_params(model, learning_rate):
    # this function can be used to set custom learning rates per module
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate},
        {'params': model.decoder.parameters(), 'lr': learning_rate},
        {'params': model.conv_text_mask_out.parameters(), 'lr': learning_rate},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate},
        {'params': model.conv_binarizer.parameters(), 'lr': learning_rate}
    ]


def callback_warmup_params(model, learning_rate):
    # Initially, only the output layer is updated ....
    print("\nBaseline, warmup, only training Branches\n")
    return [
        {'params': model.conv_text_mask_out.parameters(), 'lr': learning_rate},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate},
        {'params': model.conv_binarizer.parameters(), 'lr': learning_rate}
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

    trainer = Stage2TextDelTrainer(config, False)

    trainer.load_datasets()
    trainer.create_data_loaders()

    DEVICE = config.get("LectureNet.General.Device", "cuda:0")

    lecture_net = FCN_LectureNet.CreateFromConfig(config, 3, False)

    trainer.load_weights(lecture_net, False)

    lecture_net = lecture_net.to(DEVICE)

    trainer.log_trainable_parameters(lecture_net)

    # Get the optimization manager (optimizer, learning rate control, check points)
    trainer.create_op_manager(lecture_net, callback_optimize_params, callback_warmup_params, None)

    if len(sys.argv) >= 3:
        trainer.load_checkpoint(sys.argv[2])

    use_compile = trainer.txt_config.get("Optimization.CompileModel")
    if use_compile:
        # Caution: might be troublesome on Windows!
        trainer.perception_loss = torch.compile(trainer.perception_loss, dynamic=True)

        # full model compilation = trouble!
        train_model = torch.compile(lecture_net, dynamic=False)
    else:
        train_model = lecture_net

    # if debug images are enabled AND it's the first epoch (do not overwrite if training from checkpoint)
    if trainer.op_manager.current_epoch() == 1:
        trainer.save_debug(0, train_model)

    trainer.exec_training(train_model)

    trainer.save_pretrained_model(lecture_net)

    end_time = time.time()
    trainer.log_overall_times(end_time - start_time)


if __name__ == "__main__":
    main()
