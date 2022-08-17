# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTNeck(BaseModule):
    """Neck for BEiT Pre-training.

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
                 num_classes: int = 8192,
                 embed_dims: int = 768,
                 init_cfg: dict = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.decoders = nn.Linear(
            embed_dims, num_classes) if num_classes > 0 else nn.Identity()

    def init_weights(self) -> None:
        super(BEiTNeck, self).init_weights()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the latent prediction and final prediction.

        Args:
            x (torch.Tensor): Features of tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final prediction.
        """

        logits = self.decoders(x)
        logits = logits.view(-1, logits.shape[-1])

        return logits
