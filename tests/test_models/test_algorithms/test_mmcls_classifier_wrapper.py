# Copyright (c) OpenMMLab. All rights reserved.
import platform

import mmcls.models  # noqa: F401
import pytest
import torch

from mmselfsup.models import ALGORITHMS


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mmcls_classifier_wrapper():
    model_config = dict(
        type='MMClsImageClassifierWrapper',
        backbone=dict(
            type='mmcls.SwinTransformer',
            arch='base',
            img_size=192,
            drop_path_rate=0.1,
            stage_cfgs=dict(block_cfgs=dict(window_size=6))),
        neck=dict(type='mmcls.GlobalAveragePooling'),
        head=dict(
            type='mmcls.LinearClsHead',
            num_classes=1000,
            in_channels=1024,
            init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
            loss=dict(
                type='mmcls.LabelSmoothLoss',
                label_smooth_val=0.1,
                mode='original'),
            cal_acc=False),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ],
        train_cfg=dict(augments=[
            dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
            dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
        ]))
    model = ALGORITHMS.build(model_config)
    fake_inputs = torch.rand((2, 3, 192, 192))
    fake_labels = torch.rand((2, 1))

    # train mode
    outputs = model(fake_inputs, label=fake_labels, mode='train')
    assert isinstance(outputs['loss'], torch.Tensor)

    # test mode
    outputs = model(fake_inputs, mode='test')
    assert list(outputs['head3'].shape) == [2, 1000]

    # extract mode
    outputs = model(fake_inputs, mode='extract')
    assert list(outputs[0].shape) == [2, 1024]

    # invalid mode
    with pytest.raises(Exception):
        outputs = model(fake_inputs, mode='invalid')
