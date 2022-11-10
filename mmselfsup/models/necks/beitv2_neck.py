# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_

from mmselfsup.models.utils import BEiTV2ClsPretrainLayers
from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTV2Neck(BaseModule):
    """Neck for BEiTV2 Pre-training.

    This module construct the decoder for the final prediction.

    Args:
    """

    def __init__(self,
                 num_layers: int = 2,
                 early_layers: int = 9,
                 embed_dims: int = 768,
                 backbone_arch: str = 'base',
                 layer_scale_init_value: float = 0.1,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.early_layers = early_layers

        self.cls_pt_layers = BEiTV2ClsPretrainLayers(
            num_layers=num_layers,
            early_layers=early_layers,
            backbone_arch=backbone_arch,
            layer_scale_init_value=layer_scale_init_value,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg)
        self.fix_init_cls_pt_weight()

        _, norm = build_norm_layer(norm_cfg, embed_dims)
        self.add_module('norm', norm)

    def fix_init_cls_pt_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.cls_pt_layers.layers):
            rescale(layer.attn.proj.weight.data,
                    self.early_layers + layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data,
                    self.early_layers + layer_id + 1)

    def init_weights(self) -> None:
        super().init_weights()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Tuple[torch.Tensor], **kwargs) -> tuple:
        """Get the latent prediction and final prediction.

        Args:
            x (torch.Tensor): Features of tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final prediction.
        """

        early_states, x = inputs[0], inputs[1]
        x_cls_pt = torch.cat([x[:, [0]], early_states[:, 1:]], dim=1)
        for blk in self.cls_pt_layers.layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=kwargs['rel_pos_bias'])

        # shared norm
        x, x_cls_pt = self.norm(x), self.norm(x_cls_pt)

        # remove cls_token
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        return x, x_cls_pt
