# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MAEReconstructionLoss(BaseModule):
    """Loss function for MAE.

    Compute the loss in masked region.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MAE Loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()

        return loss
