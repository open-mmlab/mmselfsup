# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from mmselfsup.core.hooks import StepFixCosineAnnealingLrUpdaterHook


def test_cosine_annealing_hook():
    lr_config = dict(
        min_lr=1e-5,
        warmup='linear',
        warmup_iters=10,
        warmup_ratio=1e-4,
        warmup_by_epoch=True,
        by_epoch=False)
    lr_annealing_hook = StepFixCosineAnnealingLrUpdaterHook(**lr_config)
    lr_annealing_hook.regular_lr = [1.0]
    lr_annealing_hook.warmup_iters = 10

    # test get_warmup_lr
    lr = lr_annealing_hook.get_warmup_lr(1)
    assert isinstance(lr, list)

    # test get_lr
    # by_epoch = False
    runner = MagicMock()
    runner.iter = 10
    runner.max_iters = 1000
    lr = lr_annealing_hook.get_lr(runner, 1.5)
    assert isinstance(lr, float)

    # by_epoch = True
    lr_annealing_hook.by_epoch = True
    runner.epoch = 10
    runner.max_epochs = 10
    runner.data_loader = MagicMock()
    runner.data_loader.__len__ = MagicMock(return_value=10)
    lr = lr_annealing_hook.get_lr(runner, 1.5)
    assert isinstance(lr, float)
