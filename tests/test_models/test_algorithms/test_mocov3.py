# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.core import SelfSupDataSample
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
    temperature=0.2)
loss = dict(type='mmcls.CrossEntropyLoss', loss_weight=2 * 0.2)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mocov3():
    preprocess_cfg = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = MoCoV3(
            backbone=None,
            neck=neck,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MoCoV3(
            backbone=backbone,
            neck=None,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MoCoV3(
            backbone=backbone,
            neck=neck,
            head=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MoCoV3(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))

    alg = MoCoV3(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))

    fake_data = [{
        'inputs': [torch.randn((3, 224, 224)),
                   torch.randn((3, 224, 224))],
        'data_sample':
        SelfSupDataSample()
    } for _ in range(2)]

    # test extract
    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_backbone_out = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert fake_backbone_out[0][0].size() == torch.Size([2, 384, 14, 14])
    assert fake_backbone_out[0][1].size() == torch.Size([2, 384])
