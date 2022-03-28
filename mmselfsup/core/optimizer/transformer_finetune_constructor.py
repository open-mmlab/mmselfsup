# Copyright (c) OpenMMLab. All rights reserved.
import re

import torch.distributed as dist
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.utils import build_from_cfg, print_log


@OPTIMIZER_BUILDERS.register_module()
class TransformerFinetuneConstructor:
    """Rewrote default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    In addition, we provide two optional parameters, ``model_type`` and
    ``layer_decay`` to set the commonly used layer-wise learning rate decay
    schedule. Currently, we only support layer-wise learning rate schedule
    for swin and vit.

    Args:
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are
                - `type`: class name of the optimizer.
            Optional fields are
                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, model_type, layer_decay, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.
            Defaults to None.


    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001, model_type='vit')
        >>> paramwise_cfg = dict('bias': dict(weight_decay=0., \
                                 lars_exclude=True))
        >>> optim_builder = TransformerFinetuneConstructor(
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
        # Choose which type of layer-wise lr decay to use. Currently, we only
        # support ViT and Swin.
        self.model_type = self.optimizer_cfg.pop('model_type', None)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        paramwise_options = self.paramwise_cfg

        # generate layer-wise lr decay
        if self.layer_decay > 0:
            if self.model_type == 'swin':
                self._generate_swin_layer_wise_lr_decay(
                    model, paramwise_options)
            elif self.model_type == 'vit':
                self._generate_vit_layer_wise_lr_decay(model,
                                                       paramwise_options)
            else:
                raise NotImplementedError(f'Currently, we do not support \
                    layer-wise lr decay for {self.model_type}')

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

    def _generate_swin_layer_wise_lr_decay(self, model, paramwise_options):
        """Generate layer-wise learning rate decay for Swin Transformer."""
        num_layers = sum(model.backbone.depths) + 2
        layer_scales = list(self.layer_decay**i
                            for i in reversed(range(num_layers)))

        for name, _ in model.named_parameters():

            layer_id = self._get_swin_layer(name, num_layers,
                                            model.backbone.depths)
            paramwise_options[name] = dict(lr_mult=layer_scales[layer_id])

    def _get_swin_layer(self, name, num_layers, depths):
        if 'mask_token' in name:
            return 0
        elif 'patch_embed' in name:
            return 0
        elif name.startswith('backbone.stages'):
            layer_id = int(name.split('.')[2])
            block_id = name.split('.')[4]
            if block_id == 'reduction' or block_id == 'norm':
                return sum(depths[:layer_id + 1])
            layer_id = sum(depths[:layer_id]) + int(block_id)
            return layer_id + 1
        else:
            return num_layers - 1

    def _generate_vit_layer_wise_lr_decay(self, model, paramwise_options):
        """Generate layer-wise learning rate decay for Vision Transformer."""
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
