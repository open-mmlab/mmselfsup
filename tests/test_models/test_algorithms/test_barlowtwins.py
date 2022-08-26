# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.models import BarlowTwins
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='NonLinearNeck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    num_layers=3,
    with_last_bn=False,
    with_last_bn_affine=False,
    with_avg_pool=True,
    norm_cfg=dict(type='BN1d'))
head = dict(
    type='LatentCrossCorrelationHead',
    in_channels=2,
    loss=dict(type='CrossCorrelationLoss'))


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_barlowtwins():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }
    alg = BarlowTwins(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_sample': [SelfSupDataSample() for _ in range(2)]
    }

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_loss['loss'].item(), float)

    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert list(fake_feats[0].shape) == [2, 512, 7, 7]
