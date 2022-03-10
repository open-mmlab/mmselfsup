# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import MagicMock

import pytest
import torch

import mmselfsup
from mmselfsup.models.algorithms import MoCo

queue_len = 32
feat_dim = 4
momentum = 0.999
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='MoCoV2Neck',
    in_channels=2048,
    hid_channels=4,
    out_channels=4,
    with_avg_pool=True)
head = dict(type='ContrastiveHead', temperature=0.2)


def mock_batch_shuffle_ddp(img):
    return img, 0


def mock_batch_unshuffle_ddp(img, mock_input):
    return img


def mock_concat_all_gather(img):
    return img


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_moco():
    with pytest.raises(AssertionError):
        alg = MoCo(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = MoCo(backbone=backbone, neck=neck, head=None)

    alg = MoCo(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum)
    assert alg.queue.size() == torch.Size([feat_dim, queue_len])

    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
    with pytest.raises(AssertionError):
        fake_backbone_out = alg.forward_train(fake_input)

    mmselfsup.models.algorithms.moco.batch_shuffle_ddp = MagicMock(
        side_effect=mock_batch_shuffle_ddp)
    mmselfsup.models.algorithms.moco.batch_unshuffle_ddp = MagicMock(
        side_effect=mock_batch_unshuffle_ddp)
    mmselfsup.models.algorithms.moco.concat_all_gather = MagicMock(
        side_effect=mock_concat_all_gather)
    fake_loss = alg.forward_train([fake_input, fake_input])
    assert fake_loss['loss'] > 0
    assert alg.queue_ptr.item() == 16
