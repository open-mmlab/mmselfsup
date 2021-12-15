# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.algorithms import DeepCluster

num_classes = 5
with_sobel = True,
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=2,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(type='AvgPool2dNeck')
head = dict(
    type='ClsHead',
    with_avg_pool=False,  # already has avgpool in the neck
    in_channels=2048,
    num_classes=num_classes)


def test_deepcluster():
    with pytest.raises(AssertionError):
        alg = DeepCluster(
            backbone=backbone, with_sobel=with_sobel, neck=neck, head=None)
    alg = DeepCluster(
        backbone=backbone, with_sobel=with_sobel, neck=neck, head=head)
    assert alg.num_classes == num_classes
    assert hasattr(alg, 'sobel_layer')
    assert hasattr(alg, 'neck')
    assert hasattr(alg, 'head')

    fake_input = torch.randn((16, 3, 224, 224))
    fake_labels = torch.ones(16, dtype=torch.long)
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
    fake_out = alg.forward_train(fake_input, fake_labels)
    assert fake_out['loss'].item() > 0
