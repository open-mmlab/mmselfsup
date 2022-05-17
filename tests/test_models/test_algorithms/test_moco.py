# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
from unittest.mock import MagicMock

import pytest
import torch

import mmselfsup
from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.algorithms import MoCo

queue_len = 32
feat_dim = 2
momentum = 0.999
backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='MoCoV2Neck',
    in_channels=512,
    hid_channels=2,
    out_channels=2,
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
    preprocess_cfg = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = MoCo(
            backbone=None,
            neck=neck,
            head=head,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MoCo(
            backbone=backbone,
            neck=None,
            head=head,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MoCo(
            backbone=backbone,
            neck=neck,
            head=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))

    alg = MoCo(
        backbone=backbone,
        neck=neck,
        head=head,
        queue_len=queue_len,
        feat_dim=feat_dim,
        momentum=momentum,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))
    assert alg.queue.size() == torch.Size([feat_dim, queue_len])

    fake_data = [{
        'inputs': [torch.randn((3, 224, 224)),
                   torch.randn((3, 224, 224))],
        'data_sample':
        SelfSupDataSample()
    } for _ in range(2)]

    mmselfsup.models.algorithms.moco.batch_shuffle_ddp = MagicMock(
        side_effect=mock_batch_shuffle_ddp)
    mmselfsup.models.algorithms.moco.batch_unshuffle_ddp = MagicMock(
        side_effect=mock_batch_unshuffle_ddp)
    mmselfsup.models.algorithms.moco.concat_all_gather = MagicMock(
        side_effect=mock_concat_all_gather)
    fake_loss = alg(fake_data, return_loss=True)
    assert fake_loss['loss'] > 0
    assert alg.queue_ptr.item() == 2

    # test extract
    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_backbone_out = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert fake_backbone_out[0].size() == torch.Size([2, 512, 7, 7])
