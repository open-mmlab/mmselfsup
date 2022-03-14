# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import ODC

num_classes = 5
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='ODCNeck',
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    norm_cfg=dict(type='BN1d'),
    with_avg_pool=True)
head = dict(
    type='ClsHead',
    with_avg_pool=False,
    in_channels=4,
    num_classes=num_classes)
memory_bank = dict(
    type='ODCMemory',
    length=8,
    feat_dim=4,
    momentum=0.5,
    num_classes=num_classes,
    min_cluster=2,
    debug=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or platform.system() == 'Windows',
    reason='CUDA is not available or Windows mem limit')
def test_odc():
    with pytest.raises(AssertionError):
        alg = ODC(backbone=backbone, neck=neck, head=head, memory_bank=None)
    with pytest.raises(AssertionError):
        alg = ODC(
            backbone=backbone, neck=neck, head=None, memory_bank=memory_bank)

    alg = ODC(backbone=backbone, neck=neck, head=head, memory_bank=memory_bank)
    alg.set_reweight()

    fake_input = torch.randn((16, 3, 224, 224))
    fake_out = alg.forward_test(fake_input)
    assert 'head0' in fake_out
    assert fake_out['head0'].size() == torch.Size([16, num_classes])
