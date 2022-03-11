# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import Classification


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_classification():
    # test ResNet
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

    alg = Classification(backbone=backbone, with_sobel=with_sobel, head=head)
    assert hasattr(alg, 'sobel_layer')
    assert hasattr(alg, 'head')

    fake_input = torch.randn((16, 3, 224, 224))
    fake_labels = torch.ones(16, dtype=torch.long)
    fake_out = alg.forward_test(fake_input)
    assert 'head4' in fake_out
    assert fake_out['head4'].size() == torch.Size([16, 4])

    fake_out = alg.forward_train(fake_input, fake_labels)
    assert fake_out['loss'].item() > 0

    # test ViT
    backbone = dict(
        type='VisionTransformer',
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True)
    head = dict(
        type='ClsHead',
        in_channels=384,
        num_classes=1000,
        vit_backbone=True,
    )

    alg = Classification(backbone=backbone, head=head)
    assert alg.with_head

    fake_input = torch.randn((16, 3, 224, 224))
    fake_labels = torch.ones(16, dtype=torch.long)
    fake_out = alg.forward_train(fake_input, fake_labels)
    assert fake_out['loss'].item() > 0
