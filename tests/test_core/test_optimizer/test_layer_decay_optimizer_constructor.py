# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcls.models import SwinTransformer
from torch import nn

from mmselfsup.core import LearningRateDecayOptimizerConstructor


class ToyViTBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1))
        self.patch_embed = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Conv2d(3, 3, 1)
            self.layers.append(layer)


class ToyViT(nn.Module):

    def __init__(self):
        super().__init__()
        # add some variables to meet unit test coverate rate
        self.backbone = ToyViTBackbone()
        self.head = nn.Linear(1, 1)


class ToySwin(nn.Module):

    def __init__(self):
        super().__init__()
        # add some variables to meet unit test coverate rate
        self.backbone = SwinTransformer()
        self.head = nn.Linear(1, 1)


expected_layer_wise_wd_lr_vit = [{
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 4
}, {
    'weight_decay': 0.0,
    'lr_scale': 4
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}]

base_lr = 1.0
base_wd = 0.05


def check_optimizer_lr_wd(optimizer, gt_lr_wd):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    assert len(param_groups) == len(gt_lr_wd)
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lr_wd[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lr_wd[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_learning_rate_decay_optimizer_constructor():
    model = ToyViT()
    optimizer_config = dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=base_wd,
        model_type='vit',
        layer_decay_rate=2.0)

    # test when model_type is None
    with pytest.raises(AssertionError):
        optimizer_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg=optimizer_config)
        optimizer_config['model_type'] = None
        optimizer = optimizer_constructor(model)

    # test when model_type is invalid
    with pytest.raises(AssertionError):
        optimizer_constructor = LearningRateDecayOptimizerConstructor(
            optimizer_cfg=optimizer_config)
        optimizer_config['model_type'] = 'invalid'
        optimizer = optimizer_constructor(model)

    # test vit
    optimizer_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg=optimizer_config)
    optimizer_config['model_type'] = 'vit'
    optimizer = optimizer_constructor(model)
    check_optimizer_lr_wd(optimizer, expected_layer_wise_wd_lr_vit)

    # test swin
    model = ToySwin()
    optimizer_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg=optimizer_config)
    optimizer_config['model_type'] = 'swin'
    optimizer = optimizer_constructor(model)
    assert optimizer.param_groups[-1]['lr_scale'] == 1.0
    assert optimizer.param_groups[-3]['lr_scale'] == 2.0
    assert optimizer.param_groups[-5]['lr_scale'] == 4.0
