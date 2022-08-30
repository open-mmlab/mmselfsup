# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms.rotation_pred import RotationPred
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
head = dict(
    type='ClsHead',
    loss=dict(type='mmcls.CrossEntropyLoss'),
    with_avg_pool=True,
    in_channels=512,
    num_classes=4)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_rotation_pred():
    data_preprocessor = dict(
        type='mmselfsup.RotationPredDataPreprocessor',
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True)
    alg = RotationPred(
        backbone=backbone,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    bach_size = 5
    fake_data = {
        'inputs': [
            0 * torch.ones((bach_size, 3, 20, 20)),
            1 * torch.ones((bach_size, 3, 20, 20)), 2 * torch.ones(
                (bach_size, 3, 20, 20)), 3 * torch.ones((bach_size, 3, 20, 20))
        ],
        'data_sample': [SelfSupDataSample() for _ in range(bach_size)]
    }

    pseudo_label = InstanceData()
    pseudo_label.rot_label = torch.tensor([0, 1, 2, 3])
    for i in range(bach_size):
        fake_data['data_sample'][i].pseudo_label = pseudo_label

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)

    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_loss['loss'].item(), float)

    fake_prediction = alg(fake_inputs, fake_data_samples, mode='predict')
    assert len(fake_prediction) == bach_size
    assert list(fake_prediction[0].pred_score.head4.shape) == [4, 4]

    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert list(fake_feats[0].shape) == [bach_size * 4, 512, 1, 1]
