# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models import BEiT
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

data_preprocessor = dict(
    type='TwoNormDataPreprocessor',
    mean=(123.675, 116.28, 103.53),
    std=(58.395, 57.12, 57.375),
    second_mean=(-20.4, -20.4, -20.4),
    second_std=(204., 204., 204.),
    bgr_to_rgb=True)

# model settings
backbone = dict(
    type='BEiTViT',
    arch='base',
    patch_size=16,
    drop_path_rate=0.1,
    final_norm=True,
    layer_scale_init_value=0.1,
)
neck = None
head = dict(
    type='BEiTV1Head',
    embed_dims=768,
    num_embed=8192,
    loss=dict(type='BEiTLoss'))
target_generator = dict(type='DALL-E')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_beitv1():
    register_all_modules()

    model = BEiT(
        backbone=backbone,
        neck=neck,
        head=head,
        target_generator=target_generator,
        data_preprocessor=data_preprocessor)

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
