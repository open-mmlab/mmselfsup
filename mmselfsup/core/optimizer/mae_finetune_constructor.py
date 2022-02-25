# Copyright (c) OpenMMLab. All rights reserved.
import re

import torch.distributed as dist
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.utils import build_from_cfg, print_log


@OPTIMIZER_BUILDERS.register_module()
class MAEFtOptimizerConstructor:
    """Rewrote default constructor for optimizers. By default each parameter
    share the same optimizer settings, and we provide an argument
    ``paramwise_cfg`` to specify parameter-wise settings and set layer-wise
    learning rate. It is a dict and may contain the following fields:

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are
                - `type`: class name of the optimizer.
            Optional fields are
                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.
            Defaults to None
        layer_decay (float): base value for layer wise learning rate decay.
            Defaults to 0.0

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict('bias': dict(weight_decay=0., \
                                 lars_exclude=True))
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
    """

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.layer_decay = self.optimizer_cfg.pop('layer_decay', 0.0)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        paramwise_options = self.paramwise_cfg

        # generate layer-wise lr decay
        if self.layer_decay > 0:
            self._generate_layer_wise_lr_decay(model, paramwise_options)

        # if no paramwise option is specified, just use the global setting
        if paramwise_options is None:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)
        else:
            assert isinstance(paramwise_options, dict)
            params = []
            for name, param in model.named_parameters():
                param_group = {'params': [param]}
                if not param.requires_grad:
                    params.append(param_group)
                    continue

                for regexp, options in paramwise_options.items():
                    if re.search(regexp, name):
                        for key, value in options.items():
                            if key.endswith('_mult'):  # is a multiplier
                                key = key[:-5]
                                assert key in optimizer_cfg, \
                                    f'{key} not in optimizer_cfg'
                                value = optimizer_cfg[key] * value
                            param_group[key] = value
                            if not dist.is_initialized() or \
                                    dist.get_rank() == 0:
                                print_log(f'paramwise_options -- \
                                    {name}: {key}={value}')

                # otherwise use the global settings
                params.append(param_group)

            optimizer_cfg['params'] = params
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

    def _generate_layer_wise_lr_decay(self, model, paramwise_options):
        """Currently, we follow the same layer-wise lr decay schedule as
        MAE."""
        num_layers = len(model.backbone.layers) + 1
        layer_scales = list(self.layer_decay**(num_layers - i)
                            for i in range(num_layers + 1))

        if 'pos_embed' in paramwise_options:
            paramwise_options['pos_embed'].update(
                dict(lr_mult=layer_scales[0]))
        else:
            paramwise_options['pos_embed'] = dict(lr_mult=layer_scales[0])

        if 'cls_token' in paramwise_options:
            paramwise_options['cls_token'].update(
                dict(lr_mult=layer_scales[0]))
        else:
            paramwise_options['cls_token'] = dict(lr_mult=layer_scales[0])

        if 'patch_embed' in paramwise_options:
            paramwise_options['patch_embed'].update(
                dict(lr_mult=layer_scales[0]))
        else:
            paramwise_options['patch_embed'] = dict(lr_mult=layer_scales[0])

        for i in range(num_layers - 1):
            paramwise_options[f'backbone\\.layers\\.{i}\\.'] = dict(
                lr_mult=layer_scales[i + 1])
