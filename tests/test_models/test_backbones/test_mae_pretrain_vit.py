# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.backbones import MAEViT

backbone = dict(arch='b', patch_size=16, mask_ratio=0.75)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae_pretrain_vit():
    mae_pretrain_backbone = MAEViT(**backbone)
    mae_pretrain_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_outputs = mae_pretrain_backbone(fake_inputs)[0]

    assert list(fake_outputs.shape) == [2, 50, 768]
