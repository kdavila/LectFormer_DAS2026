
import os
import json
import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from .fixed_schedule import FixedSchedule
from .dynamic_schedule import DynamicSchedule
from ..configuration.configuration import Configuration

class OptimizationManager:
    def __init__(self, model, optimizer_config, schedule, loss_ema_enabled, loss_ema_alpha, save_checkpoints,
                 checkpoint_prefix, checkpoints_frequency, output_dir, verbose=True, parameters_callback=None,
                 warmup_callback=None):

        self.__model = model

        self.__optimizer_config = optimizer_config
        self.__parameters_callback = parameters_callback

        self.__schedule = schedule

        self.__loss_ema_enabled = loss_ema_enabled
        self.__loss_ema_alpha = loss_ema_alpha
        self.__loss_ema_average = None

        self.__warmup_callback = warmup_callback

        self.__current_epoch = 1

        self.__current_lr = self.__schedule.init_lr

        self.__minimum_val_loss = float('inf')

        self.__save_checkpoints = save_checkpoints
        self.__checkpoints_prefix = checkpoint_prefix
        self.__checkpoints_frequency = checkpoints_frequency
        self.__output_dir = output_dir
        self.__verbose = verbose
        self.__early_stopping = False

        self.__last_improved = False

        # TODO: if we knew that checkpoint will be loaded afterward, then we could avoid creating 2 optimizers when
        #       the optimizer uses warm-up but it is already over
        if self.__schedule.warmup_epochs == 0 or self.__schedule.warmup_type == "none":
            # simple training, start with regular optimizer ...
            self.__optimizer = OptimizationManager.GetOptimizer(model, self.__current_lr, optimizer_config,
                                                                parameters_callback)
        else:
            # advanced training, use custom optimizer depending on the type of warmup
            if self.__schedule.warmup_type == "callback":
                # simple warmup, creates a custom optimizer once, runs for a few epochs,
                # and then it switches to regular schedule
                self.__optimizer = OptimizationManager.GetOptimizer(model, self.__current_lr, optimizer_config,
                                                                    warmup_callback)
            else:
                raise Exception(f"Warmup Type <{self.__schedule.warmup_type} not Implemented!")

    def max_epochs(self):
        return self.__schedule.max_epochs

    def current_epoch(self):
        return self.__current_epoch

    def early_stopping(self):
        return self.__early_stopping

    def last_improved(self):
        return self.__last_improved

    def optimizer(self):
        return self.__optimizer

    def min_failures_until_lr_reduction(self):
        if isinstance(self.__schedule, DynamicSchedule):
            return self.__schedule.max_failures - self.__schedule.current_fails
        else:
            raise Exception("Method only supported when using dynamic schedules")

    def max_failures(self):
        if isinstance(self.__schedule, DynamicSchedule):
            return self.__schedule.max_failures
        else:
            raise Exception("Method only supported when using dynamic schedules")

    def current_failures(self):
        if isinstance(self.__schedule, DynamicSchedule):
            return self.__schedule.current_fails
        else:
            raise Exception("Method only supported when using dynamic schedules")

    def reset_failures(self):
        if isinstance(self.__schedule, DynamicSchedule):
            self.__schedule.current_fails = 0
        else:
            raise Exception("Method only supported when using dynamic schedules")

    def reset_min_loss(self):
        self.__minimum_val_loss = float('inf')
        self.__loss_ema_average = None

    def is_warming_up(self):
        if self.__schedule.warmup_epochs == 0 or self.__schedule.warmup_type == "none":
            return False
        else:
            return self.__current_epoch <= self.__schedule.warmup_epochs

    def _EMA_loss_update(self, val_loss):
        pre_val_loss = val_loss
        if self.__loss_ema_average is None:
            # register and do not update ...
            self.__loss_ema_average = val_loss
        else:
            self.__loss_ema_average = (self.__loss_ema_average * self.__loss_ema_alpha +
                                       val_loss * (1.0 - self.__loss_ema_alpha))
            # replace the value with moving average so everything from here on uses this smoothed value ...
            val_loss = self.__loss_ema_average

        if self.__verbose:
            print(f"-> EMA Enabled: Raw Loss ={pre_val_loss:.6f}, Recorded EMA Loss={self.__loss_ema_average:.6f}")

        return val_loss

    def _update_after_epoch_warmup(self, checkpoint_data, val_loss):
        # this one works the same for both dynamic and fixed schedules
        # (in fact, currently warmups always follow a fixed schedule!)
        checkpoint_data["lr_reduced"] = False
        if val_loss < self.__minimum_val_loss:
            # improved ...
            if self.__verbose:
                print(f'Warm-up: Validation loss decreased from {self.__minimum_val_loss:.6f} to {val_loss:.6f}')
            self.__last_improved = True
            checkpoint_data["improved"] = True
            self.__minimum_val_loss = val_loss
        else:
            # did not improve
            self.__last_improved = False
            checkpoint_data["improved"] = False

        if self.__schedule.warmup_epochs == self.__current_epoch:
            # warmup ends ... start regular optimization
            # discard current optimizer
            self.__optimizer = None
            # and then force cuda to empty cache (if running on GPU)
            if next(self.__model.parameters()).device.type != "cpu":
                torch.cuda.empty_cache()

            # create a new optimizer ... (with the reduced learning rate)
            self.__optimizer = self.GetOptimizer(self.__model, self.__current_lr, self.__optimizer_config,
                                                 self.__parameters_callback)
        else:
            # not done, but for other types of warm-up, something might need to be done after every epoch
            # for example, adjusting the learning rate ...
            pass

    def _update_after_epoch_dynamic_schedule(self, checkpoint_data, val_loss):
        if val_loss < self.__minimum_val_loss:
            # improved ...
            if self.__verbose:
                print(f'Validation loss decreased from {self.__minimum_val_loss:.6f} to {val_loss:.6f}')
            self.__last_improved = True

            checkpoint_data["lr_reduced"] = False
            self.__minimum_val_loss = val_loss
            self.__schedule.register_success()
        else:
            # did not improve
            self.__last_improved = False
            reduced_lr, early_stopping = self.__schedule.register_failure()
            checkpoint_data["lr_reduced"] = reduced_lr

            if reduced_lr:
                if early_stopping:
                    if self.__verbose:
                        print("Loss hasn't decrease in a while! Saving Last Model")
                    self.__early_stopping = True
                else:
                    if self.__verbose:
                        print("Loss hasn't decrease in a while! .. Reducing Learning Rate!")

                    # effectively reduce the learning rate
                    self.__current_lr *= self.__schedule.lr_reduction_ratio

                    # TODO: replace this -->
                    # discard current optimizer
                    self.__optimizer = None
                    # and then force cuda to empty cache (if running on GPU)
                    if next(self.__model.parameters()).device.type != "cpu":
                        torch.cuda.empty_cache()

                    # create a new optimizer ... (with the reduced learning rate)
                    self.__optimizer = self.GetOptimizer(self.__model, self.__current_lr, self.__optimizer_config,
                                                         self.__parameters_callback)
                    # <---
                    # TODO: with this
                    """
                    for param_group in self.__optimizer.param_groups:
                        param_group["lr"] *= self.__schedule.lr_reduction_ratio
                    """

        checkpoint_data["improved"] = self.__last_improved

    def _update_after_epoch_fixed_schedule(self, checkpoint_data, val_loss):
        # always check if loss reduced ... but this does not affect the schedule!
        if val_loss < self.__minimum_val_loss:
            self.__last_improved = True
            # improved ...
            if self.__verbose:
                print(f'Validation loss decreased from {self.__minimum_val_loss:.6f} to {val_loss:.6f}')
            self.__minimum_val_loss = val_loss
        else:
            # did not improve
            self.__last_improved = False

        checkpoint_data["improved"] = self.__last_improved

        change_lr, new_lr = self.__schedule.check_stage_change(self.__current_epoch)
        checkpoint_data["lr_reduced"] = change_lr
        if change_lr:
            if new_lr is None:
                if self.__verbose:
                    print("End of schedule has been reached! Saving Last Model")
                # outer cycle on caller might not need this
                self.__early_stopping = True
            else:
                if self.__verbose:
                    print(f"Reaching a Scheduled change of Learning Rate! LR = {new_lr}")

                # effectively reduce the learning rate
                self.__current_lr = new_lr

                # discard current optimizer
                self.__optimizer = None
                # and then force cuda to empty cache (if running on GPU)
                if next(self.__model.parameters()).device.type != "cpu":
                    torch.cuda.empty_cache()

                # create a new optimizer ... (with the reduced learning rate)
                self.__optimizer = self.GetOptimizer(self.__model, self.__current_lr, self.__optimizer_config,
                                                     self.__parameters_callback)

    def epoch_update(self, val_loss, checkpoint_add=None):
        checkpoint_data = {}

        # check if it is warming up ...
        warming_up = self.is_warming_up()

        # check if validation loss should be smoothed using EMA
        if self.__loss_ema_enabled:
            val_loss = self._EMA_loss_update(val_loss)

        if warming_up:
            # special case for warm-up epochs ...
            self._update_after_epoch_warmup(checkpoint_data, val_loss)
        else:
            if isinstance(self.__schedule, DynamicSchedule):
                self._update_after_epoch_dynamic_schedule(checkpoint_data, val_loss)
            else:
                # assume a fixed schedule is given
                self._update_after_epoch_fixed_schedule(checkpoint_data, val_loss)

        # move to the next epoch ...
        self.__current_epoch += 1

        if self.__save_checkpoints and ((self.__current_epoch - 1) % self.__checkpoints_frequency) == 0:
            common_prefix = f"{self.__output_dir}/{self.__checkpoints_prefix}"
            model_filename = f"{common_prefix}_model_{self.__current_epoch - 1}.dat"
            optim_filename = f"{common_prefix}_optim_{self.__current_epoch - 1}.dat"
            info_filename = f"{common_prefix}_info_{self.__current_epoch - 1}.json"
            checkpoint_data["learning_rate"] = self.__current_lr
            checkpoint_data["next_epoch"] = self.__current_epoch
            checkpoint_data["min_valid_loss"] = self.__minimum_val_loss
            checkpoint_data["model_filename"] = model_filename
            checkpoint_data["optimizer_filename"] = optim_filename

            # only dynamic scheduling needs to save (and restore) state ...
            if isinstance(self.__schedule, DynamicSchedule):
                checkpoint_data["current_lr_reductions"] = self.__schedule.current_lr_reductions
                checkpoint_data["current_fails"] = self.__schedule.current_fails

            # add other additional metadata
            if checkpoint_add is not None:
                checkpoint_data.update(checkpoint_add)

            # save model ...
            torch.save(self.__model.state_dict(), model_filename)

            # save optimizer ...
            torch.save(self.__optimizer.state_dict(), optim_filename)

            # save checkpoint meta-data
            with open(info_filename, "w") as out_file:
                json.dump(checkpoint_data, out_file, indent=4)

    def reset_optimizer(self):
        self.__optimizer = self.GetOptimizer(self.__model, self.__current_lr, self.__optimizer_config,
                                             self.__parameters_callback)

    def load_checkpoint(self, json_filename, checkpoint_load=None):
        with open(json_filename, "r") as in_file:
            checkpoint_data = json.load(in_file)

        self.__current_lr = checkpoint_data["learning_rate"]
        self.__current_epoch = checkpoint_data["next_epoch"]
        self.__minimum_val_loss = checkpoint_data["min_valid_loss"]

        # only dynamic scheduling needs to restore (and save) state ...
        if isinstance(self.__schedule, DynamicSchedule):
            self.__schedule.current_lr_reductions = checkpoint_data["current_lr_reductions"]
            self.__schedule.current_fails = checkpoint_data["current_fails"]

        model_filename = checkpoint_data["model_filename"]
        optim_filename = checkpoint_data["optimizer_filename"]

        # restore additional metadata as requested (assuming it was previously saved)
        if checkpoint_load is not None:
            for key in checkpoint_load:
                checkpoint_load[key] = checkpoint_data[key]

        # load the model and the optimizer parameters ...
        self.__model.load_state_dict(torch.load(model_filename, map_location="cpu"))

        if self.__schedule.warmup_type != "none" and self.__current_epoch > self.__schedule.warmup_epochs:
            # this optimizer uses warm-up, but the warmup is complete
            # must create a compatible optimizer before loading the state ...
            self.__optimizer = OptimizationManager.GetOptimizer(self.__model, checkpoint_data["learning_rate"],
                                                                self.__optimizer_config, self.__parameters_callback)

        if os.path.exists(optim_filename):
            self.__optimizer.load_state_dict(torch.load(optim_filename, map_location="cpu"))
        else:
            print(f"WARNING: could not find the file {optim_filename}")
            print("Training will continue with new optimizer")

    @staticmethod
    def GetOptimizer(model, learning_rate, optimizer_config, params_callback=None):
        if params_callback is None:
            params_to_optimize = model.parameters()
        else:
            params_to_optimize = params_callback(model, learning_rate)

        algorithm = optimizer_config.get("Algorithm", "NONE")
        print(f" -> Creating optimizer: {algorithm}")
        if algorithm == "SGD":
            momentum = optimizer_config.get("Momentum", 0.0)
            optimizer = optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)
        elif algorithm == "ADAM":
            betas = optimizer_config.get("Betas")
            optimizer = optim.Adam(params_to_optimize, lr=learning_rate, betas=betas)
        elif algorithm == "ADAMW":
            betas = optimizer_config.get("Betas")
            weight_decay = optimizer_config.get("WeightDecay")

            optimizer = optim.AdamW(params_to_optimize, lr=learning_rate, betas=betas, weight_decay=weight_decay)
        else:
            raise Exception(f"Optimization algorithm {algorithm} not supported")

        return optimizer

    @staticmethod
    def FromConfiguration(model, optimization_config, output_dir, verbose=True, parameters_callback=None,
                          warmup_callback=None, fixed_schedule=None):
        assert isinstance(optimization_config, Configuration)

        if fixed_schedule is None:
            schedule = DynamicSchedule.FromConfig(optimization_config)
        else:
            schedule = fixed_schedule

        if optimization_config.contains("LossEMA"):
            loss_ema_enabled = optimization_config.get("LossEMA.Enabled")
            loss_ema_alpha = optimization_config.get("LossEMA.Alpha")
        else:
            # not used by default
            loss_ema_enabled = False
            loss_ema_alpha = None

        optimizer_config = optimization_config.get_subconfig("Optimizer")

        checkpoints_save = optimization_config.get("CheckPoints.Save")
        checkpoints_prefix = optimization_config.get("CheckPoints.Prefix")
        if optimization_config.contains("CheckPoints.Frequency"):
            checkpoints_frequency = optimization_config.get("CheckPoints.Frequency")
        else:
            checkpoints_frequency = 1

        return OptimizationManager(model, optimizer_config, schedule, loss_ema_enabled, loss_ema_alpha,
                                   checkpoints_save, checkpoints_prefix, checkpoints_frequency, output_dir, verbose,
                                   parameters_callback, warmup_callback)

