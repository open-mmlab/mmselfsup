# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.core.data_structures.selfsup_data_sample import \
    SelfSupDataSample
from mmselfsup.models.algorithms.simclr import SimCLR

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='NonLinearNeck',  # SimCLR non-linear neck
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    num_layers=2,
    with_avg_pool=True)
head = dict(type='ContrastiveHead', temperature=0.1)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simclr():
    preprocess_cfg = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = SimCLR(
            backbone=backbone,
            neck=None,
            head=head,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = SimCLR(
            backbone=backbone,
            neck=neck,
            head=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg = SimCLR(
        backbone=backbone,
        neck=neck,
        head=head,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))

    fake_data = [{
        'inputs': [torch.randn((3, 224, 224)),
                   torch.randn((3, 224, 224))],
        'data_sample':
        SelfSupDataSample()
    } for _ in range(2)]

    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_feat = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert list(fake_feat[0].shape) == [2, 512, 7, 7]
