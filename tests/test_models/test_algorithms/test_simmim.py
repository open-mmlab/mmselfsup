# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

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
        head=dict(type='SimMIMHead', patch_size=4, encoder_in_channels=3))

    model = SimMIM(**model_config)
    fake_inputs = torch.rand((2, 3, 192, 192))
    fake_masks = torch.rand((2, 48, 48))
    outputs = model.forward_train([fake_inputs, fake_masks])
    assert isinstance(outputs['loss'], torch.Tensor)
