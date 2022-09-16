# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms import ODC
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

num_classes = 5
backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='ODCNeck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    norm_cfg=dict(type='BN1d'),
    with_avg_pool=True)
head = dict(
    type='ClsHead',
    loss=dict(type='mmcls.CrossEntropyLoss'),
    with_avg_pool=False,
    in_channels=2,
    num_classes=num_classes)
memory_bank = dict(
    type='ODCMemory',
    length=8,
    feat_dim=2,
    momentum=0.5,
    num_classes=num_classes,
    min_cluster=2,
    debug=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or platform.system() == 'Windows',
    reason='CUDA is not available or Windows mem limit')
def test_odc():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }

    alg = ODC(
        backbone=backbone,
        neck=neck,
        head=head,
        memory_bank=memory_bank,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    fake_data_sample = SelfSupDataSample()
    fake_sample_idx = InstanceData(value=torch.tensor([0]))
    fake_data_sample.sample_idx = fake_sample_idx
    fake_data = {
        'inputs': [torch.randn((2, 3, 224, 224))],
        'data_samples': [fake_data_sample for _ in range(2)]
    }

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)

    # test extract
    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert fake_feats[0].size() == torch.Size([2, 512, 7, 7])
