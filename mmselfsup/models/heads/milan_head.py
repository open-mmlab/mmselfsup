# Copyright (c) OpenMMLab. All rights reserved.
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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted features, of shape (N, L, D).
            target (torch.Tensor): Target features, of shape (N, L, D).

        Returns:
            torch.Tensor: the reconstructed loss.
        """
        loss = self.loss(pred, target)
        return loss
