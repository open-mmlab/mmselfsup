# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.models import MoCoV3
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

backbone = dict(
    type='MoCoV3ViT',
    arch='mocov3-small',  # embed_dim = 384
    img_size=224,
    patch_size=16,
    stop_grad_conv1=True)
neck = dict(
    type='NonLinearNeck',
    in_channels=384,
    hid_channels=2,
    out_channels=2,
    num_layers=2,
    with_bias=False,
    with_last_bn=True,
    with_last_bn_affine=False,
    with_last_bias=False,
    with_avg_pool=False,
    vit_backbone=True,
    norm_cfg=dict(type='BN1d'))
head = dict(
    type='MoCoV3Head',
    predictor=dict(
        type='NonLinearNeck',
        in_channels=2,
        hid_channels=2,
        out_channels=2,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False,
        norm_cfg=dict(type='BN1d')),
    loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=2 * 0.2),
    temperature=0.2)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mocov3():
    data_preprocessor = dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True)
    alg = MoCoV3(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_sample': [SelfSupDataSample() for _ in range(2)]
    }

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert fake_loss['loss'] > 0

    # test extract
    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert fake_feats[0][0].size() == torch.Size([2, 384, 14, 14])
    assert fake_feats[0][1].size() == torch.Size([2, 384])
