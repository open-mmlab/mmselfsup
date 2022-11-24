# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcls.models import SwinTransformer
from torch import nn

from mmselfsup.engine import LearningRateDecayOptimWrapperConstructor


class ToyViTBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1))
        self.pos_embed = nn.Parameter(torch.ones(1))
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


def check_optimizer_lr_wd(optimizer_wrapper, gt_lr_wd):
    assert isinstance(optimizer_wrapper.optimizer, torch.optim.AdamW)
    assert optimizer_wrapper.optimizer.defaults['lr'] == base_lr
    assert optimizer_wrapper.optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer_wrapper.optimizer.param_groups
    assert len(param_groups) == len(gt_lr_wd)
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lr_wd[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lr_wd[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_learning_rate_decay_optimizer_wrapper_constructor():
    model = ToyViT()
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=dict(
            type='AdamW',
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=base_wd,
            model_type='vit',
            layer_decay_rate=2.0))
    paramwise_cfg = dict(
        custom_keys={
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
        })

    # test when model_type is None
    with pytest.raises(AssertionError):
        optimizer_wrapper_constructor = LearningRateDecayOptimWrapperConstructor(  # noqa
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg)
        optim_wrapper_cfg['optimizer']['model_type'] = None
        optimizer_wrapper = optimizer_wrapper_constructor(model)

    # test when model_type is invalid
    with pytest.raises(AssertionError):
        optimizer_wrapper_constructor = LearningRateDecayOptimWrapperConstructor(  # noqa
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg)
        optim_wrapper_cfg['optimizer']['model_type'] = 'invalid'
        optimizer_wrapper = optimizer_wrapper_constructor(model)

    # test vit
    optimizer_wrapper_constructor = LearningRateDecayOptimWrapperConstructor(
        optim_wrapper_cfg=optim_wrapper_cfg, paramwise_cfg=paramwise_cfg)
    optim_wrapper_cfg['optimizer']['model_type'] = 'vit'
    optimizer_wrapper = optimizer_wrapper_constructor(model)
    check_optimizer_lr_wd(optimizer_wrapper, expected_layer_wise_wd_lr_vit)

    # test swin
    paramwise_cfg = dict(
        custom_keys={
            '.norm': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })
    model = ToySwin()
    optimizer_wrapper_constructor = LearningRateDecayOptimWrapperConstructor(
        optim_wrapper_cfg=optim_wrapper_cfg, paramwise_cfg=paramwise_cfg)
    optim_wrapper_cfg['optimizer']['model_type'] = 'swin'
    optimizer_wrapper = optimizer_wrapper_constructor(model)
    assert optimizer_wrapper.optimizer.param_groups[-1]['lr_scale'] == 1.0
    assert optimizer_wrapper.optimizer.param_groups[-3]['lr_scale'] == 2.0
    assert optimizer_wrapper.optimizer.param_groups[-5]['lr_scale'] == 4.0
    # check relative pos bias table is not decayed
    assert optimizer_wrapper.optimizer.param_groups[-4][
        'weight_decay'] == 0.0 and 'backbone.stages.3.blocks.1.attn.w_msa.relative_position_bias_table' in optimizer_wrapper.optimizer.param_groups[  # noqa
            -4]['param_names']
