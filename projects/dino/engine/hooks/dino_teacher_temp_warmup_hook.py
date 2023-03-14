# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from mmselfsup.registry import HOOKS
import numpy as np


@HOOKS.register_module()
class DINOTeacherTempWarmupHook(Hook):

    def __init__(self, warmup_teacher_temp: float, teacher_temp: float,
                 teacher_temp_warmup_epochs: int, max_epochs: int) -> None:
        super().__init__()
        self.teacher_temps = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp,
                         teacher_temp_warmup_epochs),
             np.ones(max_epochs - teacher_temp_warmup_epochs) * teacher_temp))

    def before_train_epoch(self, runner) -> None:
        runner.model.module.head.teacher_temp = self.teacher_temps[
            runner.epoch]
