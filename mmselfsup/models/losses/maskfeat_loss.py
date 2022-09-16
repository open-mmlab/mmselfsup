# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MaskFeatReconstructionLoss(BaseModule):
    """Loss function for MaskFeat.

    Compute the loss in masked region.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MaskFeat Loss.

        Args:
            pred (torch.Tensor): Predictions, which is of shape B x L x C.
            target (torch.Tensor): Hog features, which is of shape B x L x C.
            mask (torch.Tensor): The mask of the hog features,
                which is of shape B x L.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        B, L, C = pred.shape
        mask = mask.unsqueeze(2).expand(B, L, C)
        pred = pred[mask]
        target = target[mask]

        loss = ((pred - target)**2).mean(-1).mean()

        return loss
