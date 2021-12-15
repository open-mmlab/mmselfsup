# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.algorithms import Classification

with_sobel = True,
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=2,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'),
    frozen_stages=4)
head = dict(
    type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=4)


def test_classification():
    alg = Classification(backbone=backbone, with_sobel=with_sobel, head=head)
    assert hasattr(alg, 'sobel_layer')
    assert hasattr(alg, 'head')

    fake_input = torch.randn((16, 3, 224, 224))
    fake_labels = torch.ones(16, dtype=torch.long)
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
    fake_out = alg.forward_train(fake_input, fake_labels)
    assert fake_out['loss'].item() > 0
