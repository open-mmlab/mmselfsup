# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms import SimMIM
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()


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
        head=dict(
            type='SimMIMHead',
            patch_size=4,
            loss=dict(type='SimMIMReconstructionLoss', encoder_in_channels=3)),
        data_preprocessor={
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'bgr_to_rgb': True
        })
    model = SimMIM(**model_config)

    # test forward_train
    fake_data_sample = SelfSupDataSample()
    fake_mask = InstanceData(value=torch.rand((48, 48)))
    fake_data_sample.mask = fake_mask
    fake_data = {
        'inputs': [torch.randn((2, 3, 192, 192))],
        'data_sample': [fake_data_sample for _ in range(2)]
    }

    fake_batch_inputs, fake_data_samples = model.data_preprocessor(fake_data)
    fake_outputs = model(fake_batch_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

    # test extract_feat
    fake_inputs, fake_data_samples = model.data_preprocessor(fake_data)
    fake_feats = model.extract_feat(fake_inputs, fake_data_samples)
    assert list(fake_feats.shape) == [2, 3, 192, 192]

    # test reconstruct
    results = model.reconstruct(fake_feats, fake_data_samples)
    assert list(results.mask.value.shape) == [2, 192, 192, 3]
    assert list(results.pred.value.shape) == [2, 192, 192, 3]
