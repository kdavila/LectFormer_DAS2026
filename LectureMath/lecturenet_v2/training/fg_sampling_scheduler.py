
from LM_Tools.configuration.configuration import Configuration


class ForegroundSamplingScheduler:
    def __init__(self, op_manager, diff_ratio, current_min_fg_prc, max_min_fg_prc,
                 increase_value, increase_patience_prc):
        self.__op_manager = op_manager
        self.__diff_ratio = diff_ratio
        self.current_min_fg_prc = current_min_fg_prc
        self.active = False
        self.__max_min_fg_prc = max_min_fg_prc
        self.__increase_value = increase_value
        self.__increase_patience_prc = increase_patience_prc
        self.__increase_epochs = int(round(op_manager.max_failures() * self.__increase_patience_prc))

    def log_settings(self, log):
        msg = f"-> Ratio of Harder Samples: {self.__diff_ratio}"
        log.to_log(msg, display=True, add_time=True)
        msg = f"-> Max. value for Minimum Percentage of Foreground Pixels in Hard Samples: {self.__max_min_fg_prc}"
        log.to_log(msg, display=True, add_time=True)
        msg = f"-> Difficulty Increases/Steps: {self.__increase_value}"
        log.to_log(msg, display=True, add_time=True)
        msg = f"-> Percentage of Patience Before Difficulty Increase: {self.__increase_patience_prc}"
        log.to_log(msg, display=True, add_time=True)

    def set_from_checkpoint(self, active, new_difficulty, train_dataset, log):
        self.active = active
        if self.active:
            # restore internal values
            self.current_min_fg_prc = new_difficulty
            # enable
            train_dataset.crop_min_fg_ratio = self.__diff_ratio
            train_dataset.crop_min_fg_prc = self.current_min_fg_prc

            log.to_log(f"-> Restored training difficulty = {self.current_min_fg_prc}", display=True, add_time=True)
        else:
            # Not yet active, do not enable
            train_dataset.crop_min_fg_ratio = None
            train_dataset.crop_min_fg_prc = None

    def update_difficulty(self, train_dataset, log):
        # if the number of failures reached the required percentage of patience (e.g. 1/4 of patience)
        # and the maximum difficulty has not been reached ... then increase it...
        if ((not self.__op_manager.is_warming_up()) and self.__op_manager.current_failures() > 0 and
                (self.__op_manager.current_failures() % self.__increase_epochs == 0) and
                (self.current_min_fg_prc < self.__max_min_fg_prc)):
            # increase difficulty
            self.active = True
            self.current_min_fg_prc += self.__increase_value
            if self.current_min_fg_prc > self.__max_min_fg_prc:
                self.current_min_fg_prc = self.__max_min_fg_prc

            train_dataset.set_fg_sampling_params(self.__diff_ratio, self.current_min_fg_prc)
            log.to_log(f"-> New training difficulty = {self.current_min_fg_prc}", display=True, add_time=True)

    @staticmethod
    def CreateFromConfig(stage_config, op_manager):
        assert isinstance(stage_config, Configuration)

        # Chance of sampling a "difficult" example
        diff_ratio = stage_config.get("Optimization.SamplingSchedule.DifficultRatio")
        # Current difficulty, set up with initial value
        diff_current_min_fg_prc = stage_config.get("Optimization.SamplingSchedule.MinDifficulty")
        # Maximum Difficulty that can be reached ...
        max_min_fg_prc = stage_config.get("Optimization.SamplingSchedule.MaxDifficulty")
        # Step or Increase in Difficulty when max patience has been reached ...
        diff_increase_value = stage_config.get("Optimization.SamplingSchedule.IncreaseValue")
        # Determines the max patience before increasing difficulty
        # this is indirectly controlled as a percentage of the overall patience (Dynamic Schedules)
        diff_increase_patience_prc = stage_config.get("Optimization.SamplingSchedule.IncreasePatiencePercentage")

        return ForegroundSamplingScheduler(
            op_manager, diff_ratio, diff_current_min_fg_prc, max_min_fg_prc,
            diff_increase_value, diff_increase_patience_prc
        )
