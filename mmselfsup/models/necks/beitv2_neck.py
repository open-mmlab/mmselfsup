# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmselfsup.models.utils import BEiTV2ClsPretrainLayers
from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTV2Neck(BaseModule):
    """Neck for BEiTV2 Pre-training.

    This module construct the decoder for the final prediction.

    Args:
        num_layers (int): Number of encoder layers of neck. Defaults to 2.
        early_layers (int): The layer index of the early output from the
            backbone. Defaults to 9.
        backbone_arch (str): Vision Transformer architecture. Defaults to base.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): The initialization value for the
            learnable scaling of attention and FFN. Defaults to 0.1.
        use_rel_pos_bias (bool): Whether to use unique relative position bias,
            if False, use shared relative position bias defined in backbone.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_layers: int = 2,
        early_layers: int = 9,
        backbone_arch: str = 'base',
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 0.1,
        use_rel_pos_bias: bool = False,
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        init_cfg: Optional[Union[dict, List[dict]]] = dict(
            type='TruncNormal', layer='Linear', std=0.02, bias=0)
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.early_layers = early_layers

        self.cls_pt_layers = BEiTV2ClsPretrainLayers(
            num_layers=num_layers,
            early_layers=early_layers,
            backbone_arch=backbone_arch,
            layer_scale_init_value=layer_scale_init_value,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_rel_pos_bias=use_rel_pos_bias,
            norm_cfg=norm_cfg)
        self.fix_init_cls_pt_weight()

        embed_dims = self.cls_pt_layers.arch_settings['embed_dims']
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

    def forward(self, inputs: Tuple[torch.Tensor], rel_pos_bias: torch.Tensor,
                **kwargs) -> tuple:
        """Get the latent prediction and final prediction.

        Args:
            x (Tuple[torch.Tensor]): Features of tokens.
            rel_pos_bias (torch.Tensor): Shared relative position bias table.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final prediction.
        """

        early_states, x = inputs[0], inputs[1]
        x_cls_pt = torch.cat([x[:, [0]], early_states[:, 1:]], dim=1)
        for blk in self.cls_pt_layers.layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)

        # shared norm
        x, x_cls_pt = self.norm(x), self.norm(x_cls_pt)

        # remove cls_token
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        return x, x_cls_pt
