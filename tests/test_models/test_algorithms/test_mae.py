# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.core.data_structures.selfsup_data_sample import \
    SelfSupDataSample
from mmselfsup.models.algorithms.mae import MAE

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
head = dict(type='MAEPretrainHead', norm_pix=False, patch_size=16)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae():
    preprocess_cfg = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_rgb': True
    }
    with pytest.raises(AssertionError):
        alg = MAE(
            backbone=backbone,
            neck=None,
            head=head,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MAE(
            backbone=backbone,
            neck=neck,
            head=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = MAE(
            backbone=None,
            neck=neck,
            head=head,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg = MAE(
        backbone=backbone,
        neck=neck,
        head=head,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg.init_weights()

    fake_data = [{
        'inputs': [torch.randn((3, 224, 224))],
        'data_sample': SelfSupDataSample()
    } for _ in range(2)]
    fake_outputs = alg(fake_data, return_loss=True)
    assert isinstance(fake_outputs['loss'].item(), float)

    fake_inputs, fake_data_samples = alg.preprocss_data(fake_data)
    fake_feat = alg.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert list(fake_feat[0].shape) == [2, 50, 768]
