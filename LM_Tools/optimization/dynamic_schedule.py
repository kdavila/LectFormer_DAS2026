
class DynamicSchedule:
    def __init__(self, max_epochs, init_learning_rate, max_failures, lr_reduction_ratio, max_lr_reductions,
                 warmup_type, warmup_epochs):

        self.max_epochs = max_epochs

        self.init_lr = init_learning_rate
        self.max_failures = max_failures
        self.lr_reduction_ratio = lr_reduction_ratio
        self.max_lr_reductions = max_lr_reductions

        self.current_fails = 0
        self.current_lr_reductions = 0

        if warmup_type is None:
            warmup_type = "None"
        self.warmup_type = warmup_type.lower()
        self.warmup_epochs = warmup_epochs

    def register_success(self):
        # resets failure counter ...
        self.current_fails = 0

    def register_failure(self):
        # increase counter ...
        self.current_fails += 1
        # check ....

        if self.current_fails >= self.max_failures:
            reduced_lr = True
            if self.current_lr_reductions >= self.max_lr_reductions:
                early_stopping = True
            else:
                early_stopping = False

                # reset fails ...
                self.current_fails = 0
                # and count the reduction ...
                self.current_lr_reductions += 1
        else:
            reduced_lr = False
            early_stopping = False

        return reduced_lr, early_stopping

    @staticmethod
    def FromConfig(optimization_config):
        lr_config = optimization_config.get_subconfig("LearningRate")

        max_epochs = optimization_config.get("MaxEpochs", 25)

        learning_rate = lr_config.get("Initial", 0.1)
        max_failures = lr_config.get("MaxFailures", 3)
        lr_reduction_ratio = lr_config.get("ReductionRatio", 0.1)
        max_reductions = lr_config.get("MaxReductions", 1)

        if lr_config.contains("Warmup"):
            warmup_type = lr_config.get("Warmup.Type")
            warmup_epochs = lr_config.get("Warmup.Epochs")
        else:
            warmup_type = "none"
            warmup_epochs = 0

        return DynamicSchedule(max_epochs, learning_rate, max_failures, lr_reduction_ratio, max_reductions,
                               warmup_type, warmup_epochs)

