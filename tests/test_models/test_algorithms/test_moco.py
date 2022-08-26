# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.models.algorithms import MoCo
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

queue_len = 32
feat_dim = 2
momentum = 0.999
backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='MoCoV2Neck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    with_avg_pool=True)
head = dict(
    type='ContrastiveHead',
    loss=dict(type='mmcls.CrossEntropyLoss'),
    temperature=0.2)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_moco():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }

    alg = MoCo(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum,
        data_preprocessor=copy.deepcopy(data_preprocessor))
    assert alg.queue.size() == torch.Size([feat_dim, queue_len])

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_sample': [SelfSupDataSample() for _ in range(2)]
    }

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert fake_loss['loss'] > 0
    assert alg.queue_ptr.item() == 2

    # test extract
    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert fake_feats[0].size() == torch.Size([2, 512, 7, 7])
