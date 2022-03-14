# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import SimSiam

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'),
    zero_init_residual=True)
neck = dict(
    type='NonLinearNeck',
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    num_layers=3,
    with_last_bn_affine=False,
    with_avg_pool=True,
    norm_cfg=dict(type='BN1d'))
head = dict(
    type='LatentPredictHead',
    predictor=dict(
        type='NonLinearNeck',
        in_channels=4,
        hid_channels=4,
        out_channels=4,
        with_avg_pool=False,
        with_last_bn=False,
        with_last_bias=True,
        norm_cfg=dict(type='BN1d')))


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simsiam():
    with pytest.raises(AssertionError):
        alg = SimSiam(backbone=backbone, neck=neck, head=None)

    alg = SimSiam(backbone=backbone, neck=neck, head=head)
    with pytest.raises(AssertionError):
        fake_input = torch.randn((16, 3, 224, 224))
        alg.forward_train(fake_input)

    fake_input = [
        torch.randn((16, 3, 224, 224)),
        torch.randn((16, 3, 224, 224))
    ]
    fake_out = alg.forward(fake_input)
    assert fake_out['loss'].item() > -1

    # test train step
    fake_outputs = alg.train_step(dict(img=fake_input), None)
    assert fake_outputs['loss'].item() > -1
    assert fake_outputs['num_samples'] == 16

    # test val step
    fake_outputs = alg.val_step(dict(img=fake_input), None)
    assert fake_outputs['loss'].item() > -1
    assert fake_outputs['num_samples'] == 16
