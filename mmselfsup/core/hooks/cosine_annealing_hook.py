# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS
from mmcv.runner.hooks.lr_updater import (CosineAnnealingLrUpdaterHook,
                                          annealing_cos)


@HOOKS.register_module()
class StepFixCosineAnnealingLrUpdaterHook(CosineAnnealingLrUpdaterHook):

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs

            # Delete warmup epochs
            if self.warmup is not None:
                progress = progress - self.warmup_iters // len(
                    runner.data_loader)
                max_progress = max_progress - self.warmup_iters // len(
                    runner.data_loader)
        else:
            progress = runner.iter
            max_progress = runner.max_iters

            # Delete warmup iters
            if self.warmup is not None:
                progress = progress - self.warmup_iters
                max_progress = max_progress - self.warmup_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        return annealing_cos(base_lr, target_lr, progress / max_progress)
