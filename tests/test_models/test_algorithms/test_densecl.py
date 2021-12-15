# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.algorithms import DenseCL

queue_len = 65536
feat_dim = 128
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
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
    with pytest.raises(AssertionError):
        fake_backbone_out = alg.forward_train(fake_input)
