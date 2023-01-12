# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MILANPretrainHead(BaseModule):
    """MILAN pretrain head.

    Args:
        loss (dict): Config of loss.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted features, of shape (N, L, D).
            target (torch.Tensor): Target features, of shape (N, L, D).
            mask (torch.Tensor): The mask of the target image of shape.

        Returns:
            torch.Tensor: the reconstructed loss.
        """
        loss = self.loss(pred, target, mask)
        return loss
