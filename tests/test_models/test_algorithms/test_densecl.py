# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import MagicMock

import pytest
import torch

import mmselfsup
from mmselfsup.models.algorithms import DenseCL

queue_len = 32
feat_dim = 4
momentum = 0.999
loss_lambda = 0.5
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='DenseCLNeck',
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    num_grid=None)
head = dict(type='ContrastiveHead', temperature=0.2)


def mock_batch_shuffle_ddp(img):
    return img, 0


def mock_batch_unshuffle_ddp(img, mock_input):
    return img


def mock_concat_all_gather(img):
    return img


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_densecl():
    with pytest.raises(AssertionError):
        alg = DenseCL(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = DenseCL(backbone=backbone, neck=neck, head=None)

    alg = DenseCL(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum,
        loss_lambda=loss_lambda)
    assert alg.queue.size() == torch.Size([feat_dim, queue_len])
    assert alg.queue2.size() == torch.Size([feat_dim, queue_len])

    fake_input = torch.randn((16, 3, 224, 224))
    with pytest.raises(AssertionError):
        fake_out = alg.forward_train(fake_input)

    fake_out = alg.forward_test(fake_input)
    assert fake_out[0] is None
    assert fake_out[2] is None
    assert fake_out[1].size() == torch.Size([16, 2048, 49])

    mmselfsup.models.algorithms.densecl.batch_shuffle_ddp = MagicMock(
        side_effect=mock_batch_shuffle_ddp)
    mmselfsup.models.algorithms.densecl.batch_unshuffle_ddp = MagicMock(
        side_effect=mock_batch_unshuffle_ddp)
    mmselfsup.models.algorithms.densecl.concat_all_gather = MagicMock(
        side_effect=mock_concat_all_gather)
    fake_loss = alg.forward_train([fake_input, fake_input])
    assert fake_loss['loss_single'] > 0
    assert fake_loss['loss_dense'] > 0
    assert alg.queue_ptr.item() == 16
    assert alg.queue2_ptr.item() == 16

    # test train step with 2 keys in loss
    fake_outputs = alg.train_step(dict(img=[fake_input, fake_input]), None)
    assert fake_outputs['loss'].item() > -1
    assert fake_outputs['num_samples'] == 16
