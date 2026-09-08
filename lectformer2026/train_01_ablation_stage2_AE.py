
import sys
sys.path.insert(0, '..')

import time

import torch
torch._dynamo.config.recompile_limit = 32

from LM_Tools.configuration.configuration import Configuration
from LM_Tools.optimization.fixed_schedule import FixedSchedule, ScheduleStage

from LectureMath.lecturenet_v2.model.lectformer import LectFormerAutoEncoder
from LectureMath.lecturenet_v2.training.stage2_textdel_trainer import Stage2TextDelTrainer


def callback_optimize_params(model, learning_rate):
    # this function can be used to set custom learning rates per module
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

    trainer = Stage2TextDelTrainer(config, True)

    trainer.load_datasets()
    trainer.create_data_loaders()

    DEVICE = config.get("LectureNet.General.Device", "cuda:0")
    use_middle_block = False  # Default.
    active_skips = config.get_subconfig("LectureNet.Network.Skips.Active").data.keys()
    use_transformer_skips = not config.get("LectureNet.Network.Skips.BypassMode")

    lecture_net = LectFormerAutoEncoder.CreateFromConfig(
        config, 3, use_middle_block, use_transformer_skips,
        0.0, None, True
    )

    trainer.load_weights(lecture_net, True)

    lecture_net = lecture_net.to(DEVICE)

    print(f"\tActive Skips: {active_skips}")
    print(f"\tUsing Middle Block: {use_middle_block}")

    trainer.log_trainable_parameters(lecture_net)

    # ... Fixed Protocol used in the paper ...
    schedule_stages = [
        ScheduleStage(0.000100, 30),
        ScheduleStage(0.000010, 15),
        ScheduleStage(0.000001, 5),
    ]
    schedule = FixedSchedule(None, 0, None, schedule_stages)

    # Get the optimization manager (optimizer, learning rate control, check points)
    trainer.create_op_manager(lecture_net, callback_optimize_params, None, schedule)

    if len(sys.argv) >= 3:
        trainer.load_checkpoint(sys.argv[2])

    use_compile = trainer.txt_config.get("Optimization.CompileModel")
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

    trainer.save_pretrained_model(lecture_net)

    end_time = time.time()
    trainer.log_overall_times(end_time - start_time)


if __name__ == "__main__":
    main()
