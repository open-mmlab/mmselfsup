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
    second_mean=(127.5, 127.5, 127.5),
    second_std=(127.5, 127.5, 127.5),
    bgr_to_rgb=True)

# model settings
vqkd_encoder = dict(
    arch='base',
    img_size=224,
    patch_size=16,
    in_channels=3,
    out_indices=-1,
    drop_rate=0.,
    drop_path_rate=0.,
    norm_cfg=dict(type='LN', eps=1e-6),
    final_norm=True,
    with_cls_token=True,
    avg_token=False,
    frozen_stages=-1,
    output_cls_token=False,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    layer_scale_init_value=0.,
    interpolate_mode='bicubic',
    patch_cfg=dict(),
    layer_cfgs=dict(),
    init_cfg=None)

layer_scale_init_value = 0.1
drop_path_rate = 0.  # 0. for 300 epochs and 0.1 for 1600 epochs.
backbone = dict(
    type='BEiTViT',
    arch='base',
    patch_size=16,
    out_indices=[-4, -1],
    drop_path_rate=drop_path_rate,
    final_norm=False,
    layer_scale_init_value=layer_scale_init_value,
)
neck = dict(
    type='BEiTV2Neck',
    num_layers=1,
    early_layers=9,
    backbone_arch='base',
    drop_path_rate=drop_path_rate,
    layer_scale_init_value=layer_scale_init_value,
)
head = dict(
    type='BEiTV2Head',
    embed_dims=768,
    num_embed=8192,
    loss=dict(type='BEiTLoss'))
target_generator = dict(type='VQKD', encoder_config=vqkd_encoder)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_beitv2():
    register_all_modules()

    model = BEiT(
        backbone=backbone,
        neck=neck,
        head=head,
        target_generator=target_generator,
        data_preprocessor=data_preprocessor)

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
    assert isinstance(fake_outputs['loss_1'].item(), float)
    assert isinstance(fake_outputs['loss_2'].item(), float)
