# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DenseCLHook(Hook):
    """Hook for DenseCL.

    This hook includes ``loss_lambda`` warmup in DenseCL.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL>`_.

    Args:
        start_iters (int, optional): The number of warmup iterations to set
            ``loss_lambda=0``. Defaults to 1000.
    """

    def __init__(self, start_iters=1000, **kwargs):
        self.start_iters = start_iters

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCL."
        self.loss_lambda = runner.model.module.loss_lambda

    def before_train_iter(self, runner):
        assert hasattr(runner.model.module, 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCL."
        cur_iter = runner.iter
        if cur_iter >= self.start_iters:
            runner.model.module.loss_lambda = self.loss_lambda
        else:
            runner.model.module.loss_lambda = 0.
