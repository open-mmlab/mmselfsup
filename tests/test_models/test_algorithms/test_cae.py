# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import CAE

# model settings
backbone = dict(
    type='CAEViT', arch='b', patch_size=16, init_values=0.1, qkv_bias=False)
neck = dict(
    type='CAENeck',
    patch_size=16,
    embed_dims=768,
    num_heads=12,
    regressor_depth=4,
    decoder_depth=4,
    mlp_ratio=4,
    init_values=0.1,
)
head = dict(
    type='CAEHead', tokenizer_path='cae_ckpt/encoder_stat_dict.pth', lambd=2)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae():
    with pytest.raises(AssertionError):
        model = CAE(backbone=None, neck=neck, head=head)
    with pytest.raises(AssertionError):
        model = CAE(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        model = CAE(backbone=backbone, neck=neck, head=None)

    model = CAE(backbone=backbone, neck=neck, head=head)
    model.init_weights()

    fake_input = torch.rand((1, 3, 224, 224))
    fake_target = torch.rand((1, 3, 112, 112))
    fake_mask = torch.zeros((1, 196)).bool()
    fake_mask[:, 75:150] = 1

    inputs = (fake_input, fake_target, fake_mask)

    fake_loss = model.forward_train(inputs)
    fake_feat = model.extract_feat(fake_input, fake_mask)
    assert isinstance(fake_loss['loss'].item(), float)
    assert list(fake_feat.shape) == [1, 122, 768]
