# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.backbones import MaskFeatViT

backbone = dict(arch='b', patch_size=16)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_maskfeat_pretrain_vit():
    maskfeat_pretrain_backbone = MaskFeatViT(**backbone)
    maskfeat_pretrain_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_mask = torch.randn((2, 14, 14))
    fake_outputs = maskfeat_pretrain_backbone(fake_inputs, fake_mask)

    assert list(fake_outputs.shape) == [2, 197, 768]
