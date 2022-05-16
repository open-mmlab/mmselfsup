# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.data import BaseDataElement as PixelData

from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.algorithms import SimMIM


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_simmim():

    # model config
    model_config = dict(
        backbone=dict(
            type='SimMIMSwinTransformer',
            arch='B',
            img_size=192,
            stage_cfgs=dict(block_cfgs=dict(window_size=6))),
        neck=dict(
            type='SimMIMNeck', in_channels=128 * 2**3, encoder_stride=32),
        head=dict(type='SimMIMHead', patch_size=4, encoder_in_channels=3),
        preprocess_cfg={
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'to_rgb': True
        })
    model = SimMIM(**model_config)

    # test forward_train
    fake_data_sample = SelfSupDataSample()
    fake_mask = PixelData(value=torch.rand((48, 48)))
    fake_data_sample.mask = fake_mask
    fake_data = [{
        'inputs': [torch.randn((3, 192, 192))],
        'data_sample': fake_data_sample
    } for _ in range(2)]
    outputs = model(fake_data, return_loss=True)
    assert isinstance(outputs['loss'], torch.Tensor)

    # test extract_feat
    fake_inputs, fake_data_samples = model.preprocss_data(fake_data)
    fake_feat = model.extract_feat(
        inputs=fake_inputs, data_samples=fake_data_samples)
    assert list(fake_feat[0].shape) == [2, 1024, 6, 6]
