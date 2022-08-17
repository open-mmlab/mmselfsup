# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.backbones import BEiTViT

backbone = dict(arch='deit-b', beit_style=True, layer_scale_init_value=0.1)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_beit_vit():
    beit_backbone = BEiTViT(**backbone)
    beit_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_mask = torch.zeros((2, 196)).bool()
    fake_mask[:, 75:150] = 1
    fake_outputs = beit_backbone(fake_inputs, fake_mask)

    assert list(fake_outputs.shape) == [2, 197, 768]
