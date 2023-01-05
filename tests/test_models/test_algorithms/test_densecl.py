# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
from unittest.mock import MagicMock

import pytest
import torch

import mmselfsup
from mmselfsup.models.algorithms.densecl import DenseCL
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

queue_len = 32
feat_dim = 2
momentum = 0.999
loss_lambda = 0.5
backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='DenseCLNeck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
    num_grid=None)
head = dict(
    type='ContrastiveHead',
    loss=dict(type='mmcls.CrossEntropyLoss'),
    temperature=0.2)


def mock_batch_shuffle_ddp(img):
    return img, 0


def mock_batch_unshuffle_ddp(img, mock_input):
    return img


def mock_concat_all_gather(img):
    return img


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_densecl():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }

    alg = DenseCL(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum,
        loss_lambda=loss_lambda,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    assert alg.queue.size() == torch.Size([feat_dim, queue_len])
    assert alg.queue2.size() == torch.Size([feat_dim, queue_len])

    mmselfsup.models.algorithms.densecl.batch_shuffle_ddp = MagicMock(
        side_effect=mock_batch_shuffle_ddp)
    mmselfsup.models.algorithms.densecl.batch_unshuffle_ddp = MagicMock(
        side_effect=mock_batch_unshuffle_ddp)
    mmselfsup.models.algorithms.densecl.concat_all_gather = MagicMock(
        side_effect=mock_concat_all_gather)

    fake_data = {
        'inputs':
        [torch.randn((2, 3, 224, 224)),
         torch.randn((2, 3, 224, 224))],
        'data_sample': [SelfSupDataSample() for _ in range(2)]
    }

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_loss['loss_single'].item(), float)
    assert isinstance(fake_loss['loss_dense'].item(), float)
    assert fake_loss['loss_single'].item() > 0
    assert fake_loss['loss_dense'].item() > 0
    assert alg.queue_ptr.item() == 2
    assert alg.queue2_ptr.item() == 2

    fake_feat = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert list(fake_feat[0].shape) == [2, 512, 7, 7]

    fake_outputs = alg(fake_inputs, fake_data_samples, mode='predict')
    assert 'q_grid' in fake_outputs
    assert 'value' in fake_outputs.q_grid
    assert list(fake_outputs.q_grid.value.shape) == [2, 512, 49]
