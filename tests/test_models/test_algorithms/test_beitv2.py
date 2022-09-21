# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms import BEiT
from mmselfsup.structures import SelfSupDataSample

# model settings
backbone = dict(
    type='BEiTViT',
    arch='base',
    patch_size=16,
    out_indices=[-4, -1],
    drop_path_rate=0.1,
    final_norm=False,
    beit_style=True,
    layer_scale_init_value=0.1,
)
neck = dict(
    type='BEiTV2Neck',
    early_layers=9,
    num_classes=8192,
    embed_dims=768,
    arch='base',
    shared_lm_head=True,
)
head = dict(
    type='BEiTHead',
    tokenizer_type='vqkd',
    tokenizer_path='work_dirs/selfsup/\
        beitv2_vit-base-p16_32xb64-amp-coslr-800e_in1k/vqkd.pkl',
    loss=dict(type='BEiTLoss'))
data_preprocessor = dict(
    type='mmselfsup.CAEDataPreprocessor',
    mean=[124, 117, 104],
    std=[59, 58, 58],
    bgr_to_rgb=True)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_beit():
    model = BEiT(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=data_preprocessor)
    # model.init_weights()

    fake_img = torch.rand((1, 3, 224, 224))
    fake_target_img = torch.rand((1, 3, 224, 224))
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
