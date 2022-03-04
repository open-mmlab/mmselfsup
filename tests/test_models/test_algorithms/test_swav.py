# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import SwAV

nmb_crops = [2, 6]
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'),
    zero_init_residual=True)
neck = dict(
    type='SwAVNeck',
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    norm_cfg=dict(type='BN1d'),
    with_avg_pool=True)
head = dict(
    type='SwAVHead',
    feat_dim=4,  # equal to neck['out_channels']
    epsilon=0.05,
    temperature=0.1,
    num_crops=nmb_crops)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_swav():
    with pytest.raises(AssertionError):
        alg = SwAV(backbone=backbone, neck=neck, head=None)
    with pytest.raises(AssertionError):
        alg = SwAV(backbone=backbone, neck=None, head=head)

    alg = SwAV(backbone=backbone, neck=neck, head=head)
    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])

    fake_input = [
        torch.randn((16, 3, 224, 224)),
        torch.randn((16, 3, 224, 224)),
        torch.randn((16, 3, 96, 96)),
        torch.randn((16, 3, 96, 96)),
        torch.randn((16, 3, 96, 96)),
        torch.randn((16, 3, 96, 96)),
        torch.randn((16, 3, 96, 96)),
        torch.randn((16, 3, 96, 96)),
    ]
    fake_out = alg.forward_train(fake_input)
    assert fake_out['loss'].item() > 0
