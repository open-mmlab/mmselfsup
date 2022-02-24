# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import BYOL

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='NonLinearNeck',
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    with_bias=True,
    with_last_bn=False,
    with_avg_pool=True,
    norm_cfg=dict(type='BN1d'))
head = dict(
    type='LatentPredictHead',
    predictor=dict(
        type='NonLinearNeck',
        in_channels=4,
        hid_channels=4,
        out_channels=4,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=False,
        norm_cfg=dict(type='BN1d')))


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_byol():
    with pytest.raises(AssertionError):
        alg = BYOL(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = BYOL(backbone=backbone, neck=neck, head=None)

    alg = BYOL(backbone=backbone, neck=neck, head=head)
    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
    with pytest.raises(AssertionError):
        fake_out = alg.forward_train(fake_input)

    fake_input = [
        torch.randn((16, 3, 224, 224)),
        torch.randn((16, 3, 224, 224))
    ]
    fake_out = alg.forward_train(fake_input)
    assert fake_out['loss'].item() > -4
