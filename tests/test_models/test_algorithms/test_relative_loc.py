# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.data import LabelData

from mmselfsup.core.data_structures.selfsup_data_sample import \
    SelfSupDataSample
from mmselfsup.models.algorithms.relative_loc import RelativeLoc

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='RelativeLocNeck',
    in_channels=512,
    out_channels=32,
    with_avg_pool=True)
head = dict(
    type='ClsHead',
    with_avg_pool=False,
    in_channels=32,
    num_classes=8,
    init_cfg=[
        dict(type='Normal', std=0.005, layer='Linear'),
        dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
    ])
loss = dict(type='mmcls.CrossEntropyLoss')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_relative_loc():
    preprocess_cfg = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = RelativeLoc(
            backbone=backbone,
            neck=None,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = RelativeLoc(
            backbone=backbone,
            neck=neck,
            head=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = RelativeLoc(
            backbone=None,
            neck=neck,
            head=head,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = RelativeLoc(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))

    alg = RelativeLoc(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg.init_weights()

    batch_size = 5
    fake_data = [{
        'inputs': [
            0 * torch.ones((3, 20, 20)), 1 * torch.ones((3, 20, 20)),
            2 * torch.ones((3, 20, 20)), 3 * torch.ones(
                (3, 20, 20)), 4 * torch.ones((3, 20, 20)), 5 * torch.ones(
                    (3, 20, 20)), 6 * torch.ones((3, 20, 20)), 7 * torch.ones(
                        (3, 20, 20)), 8 * torch.ones((3, 20, 20))
        ],
        'data_sample':
        SelfSupDataSample()
    } for _ in range(batch_size)]

    patch_label = LabelData()
    patch_label.value = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    for i in range(batch_size):
        fake_data[i]['data_sample'].patch_label = patch_label

    fake_outputs = alg(fake_data, return_loss=True)
    assert isinstance(fake_outputs['loss'].item(), float)

    test_results = alg(fake_data, return_loss=False)
    assert len(test_results) == len(fake_data)
    assert list(test_results[0].prediction.head4.shape) == [8, 8]

    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_feat = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert list(fake_feat[0].shape) == [batch_size * 8, 512, 1, 1]
