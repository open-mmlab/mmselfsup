# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models import MoCoV3

backbone = dict(
    type='VisionTransformer',
    arch='mocov3-small',  # embed_dim = 384
    img_size=224,
    patch_size=16,
    stop_grad_conv1=True)
neck = dict(
    type='NonLinearNeck',
    in_channels=384,
    hid_channels=8,
    out_channels=8,
    num_layers=2,
    with_bias=False,
    with_last_bn=True,
    with_last_bn_affine=False,
    with_last_bias=False,
    with_avg_pool=False,
    vit_backbone=True)
head = dict(
    type='MoCoV3Head',
    predictor=dict(
        type='NonLinearNeck',
        in_channels=8,
        hid_channels=8,
        out_channels=8,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False),
    temperature=0.2)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mocov3():
    with pytest.raises(AssertionError):
        alg = MoCoV3(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = MoCoV3(backbone=backbone, neck=neck, head=None)

    alg = MoCoV3(backbone, neck, head)
    alg.init_weights()
    alg.momentum_update()

    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.forward(fake_input, mode='extract')
    assert fake_backbone_out[0][0].size() == torch.Size([16, 384, 14, 14])
    assert fake_backbone_out[0][1].size() == torch.Size([16, 384])
