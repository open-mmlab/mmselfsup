# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms import CAE
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

# model settings
backbone = dict(type='CAEViT', arch='b', patch_size=16, init_values=0.1)
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
    type='CAEHead',
    tokenizer_path='cae_ckpt/encoder_stat_dict.pth',
    loss=dict(type='CAELoss', lambd=2))

data_preprocessor = dict(
    type='mmselfsup.CAEDataPreprocessor',
    mean=[124, 117, 104],
    std=[59, 58, 58],
    bgr_to_rgb=True)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae():
    model = CAE(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)
    # model.init_weights()

    fake_img = torch.rand((1, 3, 224, 224))
    fake_target_img = torch.rand((1, 3, 112, 112))
    fake_mask = torch.zeros((196)).bool()
    fake_mask[75:150] = 1
    fake_data_sample = SelfSupDataSample()
    fake_mask = InstanceData(value=fake_mask)
    fake_data_sample.mask = fake_mask
    fake_data_sample = [fake_data_sample]

    fake_data = {
        'inputs': [fake_img, fake_target_img],
        'data_sample': fake_data_sample
    }

    fake_batch_inputs, fake_data_samples = model.data_preprocessor(fake_data)
    fake_outputs = model(fake_batch_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)
