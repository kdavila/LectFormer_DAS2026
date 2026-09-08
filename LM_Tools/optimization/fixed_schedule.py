
class ScheduleStage:
    def __init__(self,stage_lr, stage_epochs):
        self.lr = stage_lr
        self.epochs = stage_epochs

class FixedSchedule:
    def __init__(self, warmup_type, warmup_epochs, warmup_lr, schedule_stages):
        # TODO: the stages should absorb warmup ..? (e.g., allow multiple rounds  of warmup)
        if warmup_type is None:
            warmup_type = "None"
        self.warmup_type = warmup_type.lower()
        self.warmup_epochs = warmup_epochs

        self.stages = schedule_stages

        if len(self.stages) == 0:
            raise Exception("Invalid configuration for fixed schedule. It must contain at least one stage")

        if warmup_type != "none" and self.warmup_epochs > 0:
            self.init_lr = warmup_lr
        else:
            self.init_lr = self.stages[0].lr

        key_epochs = [(warmup_epochs, self.stages[0])]
        total_epochs = warmup_epochs
        for stg_idx, stage in enumerate(self.stages):
            if stage.epochs < 1:
                raise Exception("Invalid configuration for fixed schedule. Each stage must have at least one epoch")

            total_epochs += stage.epochs
            if stg_idx + 1 < len(self.stages):
                next_stage = self.stages[stg_idx + 1]
            else:
                next_stage = None

            key_epochs.append((total_epochs, next_stage))

        self.max_epochs = total_epochs
        self.key_epochs = key_epochs

    def check_stage_change(self, epoch_ending):
        for epoch_change, stage in self.key_epochs:
            if epoch_change == epoch_ending:
                if stage is None:
                    # end training ...
                    # (change=True, New LR = None)
                    return True, None
                else:
                    # continue ... with new learning
                    return True, stage.lr
        # continue ... same learning rate
        return False, None
