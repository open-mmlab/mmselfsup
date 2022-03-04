# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import SimCLR

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='NonLinearNeck',  # SimCLR non-linear neck
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    num_layers=2,
    with_avg_pool=True)
head = dict(type='ContrastiveHead', temperature=0.1)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simclr():
    with pytest.raises(AssertionError):
        alg = SimCLR(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = SimCLR(backbone=backbone, neck=neck, head=None)

    alg = SimCLR(backbone=backbone, neck=neck, head=head)
    with pytest.raises(AssertionError):
        fake_input = torch.randn((16, 3, 224, 224))
        alg.forward_train(fake_input)

    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
