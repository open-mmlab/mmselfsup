# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MILANReconstructionLoss(BaseModule):
    """Loss function for MILAN.

    Compute the loss in masked region.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        loss = 2 - 2 * (pred * target).sum(dim=-1)
        loss = loss.mean()

        return loss
