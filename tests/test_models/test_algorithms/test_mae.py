# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.algorithms import MAE

backbone = dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75)
neck = dict(
    type='MAEPretrainDecoder',
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.,
)
head = dict(type='MAEPretrainHead', norm_pix_loss=False, patch_size=16)


def test_simclr():
    with pytest.raises(AssertionError):
        alg = MAE(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = MAE(backbone=backbone, neck=neck, head=None)
    with pytest.raises(AssertionError):
        alg = MAE(backbone=None, neck=neck, head=head)

    alg = MAE(backbone=backbone, neck=neck, head=head)

    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
