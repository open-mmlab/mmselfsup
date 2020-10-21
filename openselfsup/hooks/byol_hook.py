from math import cos, pi
from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper

from .registry import HOOKS


@HOOKS.register_module
class BYOLHook(Hook):
    """Hook for BYOL.

    This hook includes momentum adjustment in BYOL following:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: total steps.

    Args:
        end_momentum (float): The final momentum coefficient
            for the target network. Default: 1.
    """

    def __init__(self, end_momentum=1., update_interval=1, **kwargs):
        self.end_momentum = end_momentum
        self.update_interval = update_interval

    def before_train_iter(self, runner):
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in BYOLHook."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in BYOLHook."
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            base_m = runner.model.module.base_momentum
            m = self.end_momentum - (self.end_momentum - base_m) * (
                cos(pi * cur_iter / float(max_iter)) + 1) / 2
            runner.model.module.momentum = m

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()
