# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List

import torch
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor
from torch import nn

from mmselfsup.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                                OPTIMIZERS)


def get_layer_id_for_vit(var_name: str, max_layer_id: int) -> int:
    """Get the layer id to set the different learning rates for ViT.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.
    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


def get_layer_id_for_swin(var_name: str, max_layer_id: int,
                          depths: List[int]) -> int:
    """Get the layer id to set the different learning rates for Swin.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.
        depths (List[int]): Depths for each stage.
    Returns:
        int: Returns the layer id of the key.
    """
    if 'mask_token' in var_name:
        return 0
    elif 'patch_embed' in var_name:
        return 0
    elif var_name.startswith('backbone.stages'):
        layer_id = int(var_name.split('.')[2])
        block_id = var_name.split('.')[4]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return max_layer_id - 1


def get_layer_id_for_mixmim(var_name: str, max_layer_id: int,
                            depths: List[int]) -> int:
    """Get the layer id to set the different learning rates for MixMIM.

    The layer is from 1 to max_layer_id (e.g. 25)
    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.
        depths (List[int]): Depths for each stage.
    Returns:
        int: Returns the layer id of the key.
    """

    if 'patch_embed' in var_name:
        return -1
    elif 'absolute_pos_embed' in var_name:
        return -1
    elif 'pos_embed' in var_name:
        return -1
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        block_id = var_name.split('.')[4]

        if block_id == 'downsample' or \
                block_id == 'reduction' or \
                block_id == 'norm':
            return sum(depths[:layer_id + 1]) - 1

        layer_id = sum(depths[:layer_id]) + int(block_id) + 1
        return layer_id - 1
    else:
        return max_layer_id - 2


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ViT and Swin.

    In addition to applying layer-wise learning rate decay schedule, the
    paramwise_cfg only supports weight decay customization.
    """

    def add_params(self, params: List[dict], module: nn.Module,
                   optimizer_cfg: dict, **kwargs) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (List[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            optimizer_cfg (dict): The configuration of optimizer.
            prefix (str): The prefix of the module.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        # get logger
        logger = MMLogger.get_current_instance()
        logger.warning(
            'LearningRateDecayOptimWrapperConstructor is refactored in '
            'v1.0.0rc4, which need to configure zero weight decay manually. '
            'The previous versions would set zero weight decay according to '
            'the dimension of parameter. Please specify weight decay settings '
            'of different layers in config if needed.')

        # Check if self.param_cfg is not None
        if len(self.paramwise_cfg) > 0:
            logger.info(
                'The paramwise_cfg only supports weight decay customization '
                'in LearningRateDecayOptimWrapperConstructor, please indicate '
                'the specific weight decay settings of different layers in '
                'config if needed.')

        model_type = optimizer_cfg.pop('model_type', None)
        # model_type should not be None
        assert model_type is not None, 'When using layer-wise learning rate \
            decay, model_type should not be None.'

        # currently, we only support layer-wise learning rate decay for vit
        # and swin.
        assert model_type in ['vit', 'swin',
                              'mixmim'], f'Currently, we do not support \
            layer-wise learning rate decay for {model_type}'

        if model_type == 'vit':
            num_layers = len(module.backbone.layers) + 2
        elif model_type == 'swin':
            num_layers = sum(module.backbone.depths) + 2
        elif model_type == 'mixmim':
            num_layers = sum(module.backbone.depths) + 1

        # if layer_decay_rate is not provided, not decay
        decay_rate = optimizer_cfg.pop('layer_decay_rate', 1.0)
        parameter_groups = {}

        assert self.base_wd is not None
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights

            this_weight_decay = self.base_wd
            for key in sorted_keys:
                if key in name:
                    decay_mult = custom_keys[key].get('decay_mult', 1.)
                    this_weight_decay = self.base_wd * decay_mult

            if this_weight_decay == 0:
                group_name = 'no_decay'
            else:
                group_name = 'decay'

            if model_type == 'vit':
                layer_id = get_layer_id_for_vit(name, num_layers)
            elif model_type == 'swin':
                layer_id = get_layer_id_for_swin(name, num_layers,
                                                 module.backbone.depths)
            elif model_type == 'mixmim':
                layer_id = get_layer_id_for_mixmim(name, num_layers,
                                                   module.backbone.depths)

            group_name = f'layer_{layer_id}_{group_name}'
            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())

    def __call__(self, model: nn.Module) -> torch.optim.Optimizer:
        """When paramwise option is None, also use layer wise learning rate
        decay."""
        if hasattr(model, 'module'):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        optimizer_cfg = self.optimizer_cfg.copy()

        # set param-wise lr and weight decay recursively
        params: List = []
        self.add_params(params, model, optimizer_cfg)
        optimizer_cfg['params'] = params

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))

        return optim_wrapper
