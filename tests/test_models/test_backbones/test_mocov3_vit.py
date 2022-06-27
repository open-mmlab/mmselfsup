# Copyright (c) OpenMMLab. All rights reserved.
from mmselfsup.models.backbones import MoCoV3ViT


def test_vision_transformer():
    vit = MoCoV3ViT(
        arch='mocov3-small', patch_size=16, frozen_stages=12, norm_eval=True)
    vit.init_weights()
    vit.train()

    for p in vit.parameters():
        assert p.requires_grad is False
