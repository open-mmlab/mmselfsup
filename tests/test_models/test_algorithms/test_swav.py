# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.models.algorithms.swav import SwAV
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

nmb_crops = [2, 6]
backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'),
    zero_init_residual=True)
neck = dict(
    type='SwAVNeck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    norm_cfg=dict(type='BN1d'),
    with_avg_pool=True)
head = dict(
    type='SwAVHead',
    loss=dict(
        type='SwAVLoss',
        feat_dim=2,  # equal to neck['out_channels']
        epsilon=0.05,
        temperature=0.1,
        num_crops=nmb_crops))


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_swav():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }

    alg = SwAV(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    fake_data = {
        'inputs': [
            torch.randn((2, 3, 224, 224)),
            torch.randn((2, 3, 224, 224)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96)),
            torch.randn((2, 3, 96, 96))
        ],
        'data_sample': [SelfSupDataSample() for _ in range(2)]
    }

    fake_batch_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_outputs = alg(fake_batch_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

    fake_feat = alg(fake_batch_inputs, fake_data_samples, mode='tensor')
    assert list(fake_feat[0].shape) == [2, 512, 7, 7]
