# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.data import LabelData

from mmselfsup.core.data_structures.selfsup_data_sample import \
    SelfSupDataSample
from mmselfsup.models.algorithms.rotation_pred import RotationPred

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
head = dict(type='ClsHead', with_avg_pool=True, in_channels=512, num_classes=4)
loss = dict(type='mmcls.CrossEntropyLoss')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_relative_loc():
    preprocess_cfg = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = RotationPred(
            backbone=backbone,
            head=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = RotationPred(
            backbone=None,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = RotationPred(
            backbone=backbone,
            head=head,
            loss=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg = RotationPred(
        backbone=backbone,
        head=head,
        loss=loss,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg.init_weights()

    bach_size = 5
    fake_data = [{
        'inputs': [
            0 * torch.ones((3, 20, 20)), 1 * torch.ones((3, 20, 20)),
            2 * torch.ones((3, 20, 20)), 3 * torch.ones((3, 20, 20))
        ],
        'data_sample':
        SelfSupDataSample()
    } for _ in range(bach_size)]

    rot_label = LabelData()
    rot_label.value = torch.tensor([0, 1, 2, 3])
    for i in range(bach_size):
        fake_data[i]['data_sample'].rot_label = rot_label

    fake_outputs = alg(fake_data, return_loss=True)
    assert isinstance(fake_outputs['loss'].item(), float)

    test_results = alg(fake_data, return_loss=False)
    assert len(test_results) == len(fake_data)
    assert list(test_results[0].prediction.head4.shape) == [4, 4]

    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_feat = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert list(fake_feat[0].shape) == [bach_size * 4, 512, 1, 1]
