# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.utils import Mixup


def test_mixup():

    params = dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000)
    fake_input = torch.rand((2, 3, 224, 224))
    fake_targets = torch.ones((2, )).long()

    mixup = Mixup(**params)
    fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)
    assert list(fake_ouputs.shape) == [2, 3, 224, 224]
    assert list(fake_targets_mixup.shape) == [2, 1000]

    params['mixup_alpha'] = 0.0
    mixup = Mixup(**params)
    fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)
    assert list(fake_ouputs.shape) == [2, 3, 224, 224]
    assert list(fake_targets_mixup.shape) == [2, 1000]

    params['mixup_alpha'] = 0.8
    params['cutmix_alpha'] = 0.0
    mixup = Mixup(**params)
    fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)
    assert list(fake_ouputs.shape) == [2, 3, 224, 224]
    assert list(fake_targets_mixup.shape) == [2, 1000]

    params['mixup_alpha'] = 0.0
    params['cutmix_alpha'] = 0.8
    mixup = Mixup(**params)
    fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)
    assert list(fake_ouputs.shape) == [2, 3, 224, 224]
    assert list(fake_targets_mixup.shape) == [2, 1000]

    with pytest.raises(AssertionError):
        params['mixup_alpha'] = 0.0
        params['cutmix_alpha'] = 0.0
        mixup = Mixup(**params)
        fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)

    params['cutmix_minmax'] = [0.6, 1.0]
    params['mixup_alpha'] = 0.0
    params['cutmix_alpha'] = 0.8
    mixup = Mixup(**params)
    fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)
    assert list(fake_ouputs.shape) == [2, 3, 224, 224]
    assert list(fake_targets_mixup.shape) == [2, 1000]

    params['cutmix_minmax'] = None
    params['prob'] = 0.0
    mixup = Mixup(**params)
    fake_ouputs, fake_targets_mixup = mixup(fake_input, fake_targets)
    assert list(fake_ouputs.shape) == [2, 3, 224, 224]
    assert list(fake_targets_mixup.shape) == [2, 1000]
