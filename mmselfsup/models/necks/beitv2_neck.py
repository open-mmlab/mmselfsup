# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_

from mmselfsup.models.utils import BEiTV2CLSPretrainLayers
from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTV2Neck(BaseModule):
    """Neck for BEiTV2 Pre-training.

    This module construct the decoder for the final prediction.

    Args:
        patch_size (int): The patch size of each token. Defaults to 16.
        num_classes (int): The number of classes for final prediction. Defaults
            to 8192.
        embed_dims (int): The embed dims of latent feature in regressor and
            decoder. Defaults to 768.
        mask_tokens_num (int): The number of mask tokens. Defaults to 75.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 early_layers: int = 9,
                 num_classes: int = 8192,
                 embed_dims: int = 768,
                 arch: str = 'base',
                 layer_scale_init_value: float = 0.1,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN'),
                 shared_lm_head: bool = True,
                 init_cfg: dict = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.early_layers = early_layers

        self.decoders = nn.Linear(
            embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        self.cls_pt_layers = BEiTV2CLSPretrainLayers(
            arch=arch,
            layer_scale_init_value=layer_scale_init_value,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg)
        self.fix_init_cls_pt_weight()

        self.shared_lm_head = shared_lm_head

        _, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module('norm1', norm1)

    def fix_init_cls_pt_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.cls_pt_layers.layers):
            rescale(layer.attn.proj.weight.data,
                    self.early_layers + layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data,
                    self.early_layers + layer_id + 1)

    def init_weights(self) -> None:
        super(BEiTV2Neck, self).init_weights()
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

        x = self.norm1(x)
        x_cls_pt = self.norm1(x_cls_pt)

        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]

        logits, logits_1 = self.decoders(x), self.decoders(x_cls_pt)
        logits = logits.view(-1, logits.shape[-1])
        logits_1 = logits.view(-1, logits_1.shape[-1])

        return logits, logits_1
