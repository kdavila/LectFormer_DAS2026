
import sys
sys.path.insert(0, '..')

import time

from LM_Tools.configuration.configuration import Configuration
from LectureMath.lecturenet_v1.FCN_lecturenet import FCN_AutoEncoder
from LectureMath.lecturenet_v2.training.stage1_rec_trainer import Stage1RecTrainer


def callback_optimize_params(model, learning_rate):
    # this function is used to set custom per-module learning rates
    # based on relative values to the main learning rate
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.decoder.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate * 1.0}
    ]


def callback_warmup_params(model, learning_rate):
    # this function is used to set custom learning rates, usually smaller learning rates on encoder
    print("Warmup for baseline model")
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate * 0.001},
        {'params': model.decoder.parameters(), 'lr': learning_rate * 0.001},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate * 0.01}
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage")
        print(f"\tpython {sys.argv[0]:s} config [chkpt_json_file]")
        print("With:")
        print("\tconfig\t\t\tPath to configuration file")
        print("\tchkpt_json_file\tOptional. Resume training from given checkpoint")
        return

    start_time = time.time()

    # read the config file
    config = Configuration.from_file(sys.argv[1], warning_mode=True)

    # This will initialize many elements relevant to the training
    trainer = Stage1RecTrainer(config)

    trainer.load_datasets()
    trainer.create_data_loaders()

    DEVICE = config.get("LectureNet.General.Device", "cuda:0")
    prc_feat_masking = config.get("LectureNet.Pretraining.Reconstruction.AutoEncoder.FeatureMasking")
    original_arch = True

    lecture_net = FCN_AutoEncoder.CreateFromConfig(config, 3, original_arch, prc_feat_masking)
    lecture_net = lecture_net.to(DEVICE)

    print(f"\tOriginal Architecture: {original_arch}")
    print(f"\tFeature Masking: {prc_feat_masking}")
    print(f"\tUsing Laplacian Targets?: {trainer.use_laplacian_pyr}")

    trainer.log_trainable_parameters(lecture_net)

    # Get the optimization manager (optimizer, learning rate control, check points)
    trainer.create_op_manager(lecture_net, callback_optimize_params, callback_warmup_params)

    if len(sys.argv) >= 3:
        trainer.load_checkpoint(sys.argv[2])

    # if debug images are enabled AND it's the first epoch (do not overwrite if training from checkpoint)
    if trainer.op_manager.current_epoch() == 1:
        trainer.save_debug(0, lecture_net)

    trainer.exec_training(lecture_net)

    trainer.save_pretrained_model(lecture_net, False)

    end_time = time.time()
    trainer.log_overall_times(end_time - start_time)


if __name__ == "__main__":
    main()
