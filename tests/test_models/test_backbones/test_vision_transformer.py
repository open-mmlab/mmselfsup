# Copyright (c) OpenMMLab. All rights reserved.
from mmselfsup.models.backbones import VisionTransformer


def test_vision_transformer():
    vit = VisionTransformer(
        arch='mocov3-small', patch_size=16, frozen_stages=12, norm_eval=True)
    vit.train()

    for p in vit.parameters():
        assert p.requires_grad is False
