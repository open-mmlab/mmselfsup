# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class ClsHead(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        loss (dict): Config of the loss.
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (Dict or List[Dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        loss: dict,
        with_avg_pool: bool = False,
        in_channels: int = 2048,
        num_classes: int = 1000,
        vit_backbone: bool = False,
        init_cfg: Optional[Union[dict, List[dict]]] = [
            dict(type='Normal', std=0.01, layer='Linear'),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super().__init__(init_cfg)
        self.loss = MODELS.build(loss)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.vit_backbone = vit_backbone

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def logits(
        self, x: Union[List[torch.Tensor],
                       Tuple[torch.Tensor]]) -> List[torch.Tensor]:
        """Get the logits before the cross_entropy loss.

        This module is used to obtain the logits before the loss.

        Args:
            x (List[Tensor] | Tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            List[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.vit_backbone:
            x = x[-1]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                f'Tensor must has 4 dims, got: {x.dim()}'
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor]],
                label: torch.Tensor) -> torch.Tensor:
        """Get the loss.

        Args:
            x (List[Tensor] | Tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).
            label (torch.Tensor): The label for cross entropy loss.

        Returns:
            torch.Tensor: The cross entropy loss.
        """
        outs = self.logits(x)
        loss = self.loss(outs[0], label)
        return loss
