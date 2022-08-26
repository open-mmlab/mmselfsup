# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms import NPID
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
    type='LinearNeck', in_channels=512, out_channels=2, with_avg_pool=True)
head = dict(
    type='ContrastiveHead',
    loss=dict(type='mmcls.CrossEntropyLoss'),
    temperature=0.07)
memory_bank = dict(type='SimpleMemory', length=8, feat_dim=2, momentum=0.5)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_npid():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }
    alg = NPID(
        backbone=backbone,
        neck=neck,
        head=head,
        memory_bank=memory_bank,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    fake_data = {
        'inputs': [torch.randn((2, 3, 224, 224))],
        'data_samples': [
            SelfSupDataSample(
                sample_idx=InstanceData(value=torch.randint(0, 7, (1, ))))
            for _ in range(2)
        ]
    }

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert fake_loss['loss'] > -1

    # test extract
    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert fake_feats[0].size() == torch.Size([2, 512, 7, 7])
