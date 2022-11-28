# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MILANReconstructionLoss(BaseModule):
    """Loss function for MILAN.

    Compute the cosine similarity loss in all these tokens.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted features, of shape (N, L, D).
            target (torch.Tensor): Target features, of shape (N, L, D).

        Returns:
            torch.Tensor: the reconstructed loss.
        """
        loss = 2 - 2 * (pred * target).sum(dim=-1)
        loss = loss.mean()

        return loss
