# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule
from torch.nn import functional as F

from mmselfsup.registry import MODELS


@MODELS.register_module()
class SimMIMReconstructionLoss(BaseModule):
    """Loss function for MAE.

    Compute the loss in masked region.

    Args:
        encoder_in_channels (int): Number of input channels for encoder.
    """

    def __init__(self, encoder_in_channels: int) -> None:
        super().__init__()
        self.encoder_in_channels = encoder_in_channels

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
        loss_rec = F.l1_loss(target, pred, reduction='none')
        loss = (loss_rec * mask).sum() / (mask.sum() +
                                          1e-5) / self.encoder_in_channels

        return loss
