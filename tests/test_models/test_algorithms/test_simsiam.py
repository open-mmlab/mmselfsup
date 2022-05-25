# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.algorithms import SimSiam

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'),
    zero_init_residual=True)
neck = dict(
    type='NonLinearNeck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    num_layers=3,
    with_last_bn_affine=False,
    with_avg_pool=True,
    norm_cfg=dict(type='BN1d'))
head = dict(
    type='LatentPredictHead',
    predictor=dict(
        type='NonLinearNeck',
        in_channels=2,
        hid_channels=2,
        out_channels=2,
        with_avg_pool=False,
        with_last_bn=False,
        with_last_bias=True,
        norm_cfg=dict(type='BN1d')))
loss = dict(type='CosineSimilarityLoss')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simsiam():
    preprocess_cfg = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = SimSiam(
            backbone=backbone,
            neck=neck,
            head=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = SimSiam(
            backbone=backbone,
            neck=None,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = SimSiam(
            backbone=None,
            neck=neck,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = SimSiam(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))

    alg = SimSiam(
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
    fake_out = alg(fake_data, return_loss=True)
    assert fake_out['loss'].item() > -1

    # test extract
    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_feat = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert fake_feat[0].size() == torch.Size([2, 512, 7, 7])
