
import sys
sys.path.insert(0, '..')

import time

from LM_Tools.configuration.configuration import Configuration
from LectureMath.lecturenet_v2.model.lectformer import LectFormerAutoEncoder
from LectureMath.lecturenet_v2.training.stage1_rec_trainer import Stage1RecTrainer


def callback_optimize_params(model, learning_rate):
    # this function is used to set custom per-module learning rates
    # based on relative values to the main learning rate
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.skips.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.decoder.parameters(), 'lr': learning_rate * 1.0},
        {'params': model.conv_reconstruct.parameters(), 'lr': learning_rate * 1.0} # 0.001
    ]


def callback_warmup_params(model, learning_rate):
    # this function is used to set custom learning rates, usually smaller learning rates on encoder
    print("Warmup")
    return [
        {'params': model.encoder.parameters(), 'lr': learning_rate * 0.001},
        {'params': model.skips.parameters(), 'lr': learning_rate * 0.001},
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
    use_middle_block = False  # Default.
    active_skips = config.get_subconfig("LectureNet.Network.Skips.Active").data.keys()
    use_transformer_skips = not config.get("LectureNet.Network.Skips.BypassMode")

    lecture_net = LectFormerAutoEncoder.CreateFromConfig(config, 3, use_middle_block, use_transformer_skips,
                                                         prc_feat_masking, trainer.laplacian_levels)
    lecture_net = lecture_net.to(DEVICE)

    print(f"\tActive Skips: {active_skips}")
    print(f"\tUsing Middle Block: {use_middle_block}")
    print(f"\tFeature Masking: {prc_feat_masking}")
    print(f"\tUsing Laplacian Targets?: {trainer.use_laplacian_pyr}")

    trainer.log_trainable_parameters(lecture_net)

    # Get the optimization manager (optimizer, learning rate control, check points)
    if use_transformer_skips:
        # customize learning rates for transformer and encoder
        trainer.create_op_manager(lecture_net, callback_optimize_params, callback_warmup_params)
    else:
        trainer.create_op_manager(lecture_net, None, None)

    if len(sys.argv) >= 3:
        trainer.load_checkpoint(sys.argv[2])

    # if debug images are enabled AND it's the first epoch (do not overwrite if training from checkpoint)
    if trainer.op_manager.current_epoch() == 1:
        trainer.save_debug(0, lecture_net)

    trainer.exec_training(lecture_net)

    trainer.save_pretrained_model(lecture_net, use_transformer_skips)

    end_time = time.time()
    trainer.log_overall_times(end_time - start_time)


if __name__ == "__main__":
    main()
