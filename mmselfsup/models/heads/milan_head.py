# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MILANPretrainHead(BaseModule):

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MAE head.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        target = self.construct_target(target)
        loss = self.loss(pred, target, mask)

        return loss
