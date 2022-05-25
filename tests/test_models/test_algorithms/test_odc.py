# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.algorithms import ODC

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
    with_avg_pool=False,
    in_channels=2,
    num_classes=num_classes)
loss = dict(type='CrossEntropyLoss')
memory_bank = dict(
    type='ODCMemory',
    length=8,
    feat_dim=2,
    momentum=0.5,
    num_classes=num_classes,
    min_cluster=2,
    debug=False)
preprocess_cfg = {
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    'to_rgb': True
}


@pytest.mark.skipif(
    not torch.cuda.is_available() or platform.system() == 'Windows',
    reason='CUDA is not available or Windows mem limit')
def test_odc():
    with pytest.raises(AssertionError):
        alg = ODC(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=loss,
            memory_bank=None,
            preprocess_cfg=preprocess_cfg)
    with pytest.raises(AssertionError):
        alg = ODC(
            backbone=backbone,
            neck=neck,
            head=None,
            memory_bank=memory_bank,
            preprocess_cfg=preprocess_cfg)
    with pytest.raises(AssertionError):
        alg = ODC(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=loss,
            memory_bank=memory_bank,
            preprocess_cfg=preprocess_cfg)

    alg = ODC(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss,
        memory_bank=memory_bank,
        preprocess_cfg=preprocess_cfg)
    alg.set_reweight()

    fake_data = [{
        'inputs': torch.randn((3, 224, 224)),
        'data_sample': SelfSupDataSample()
    } for _ in range(2)]
    fake_out = alg(fake_data, return_loss=False)
    assert hasattr(fake_out[0].prediction, 'head0')
    assert fake_out[0].prediction.head0.size() == torch.Size([num_classes])
