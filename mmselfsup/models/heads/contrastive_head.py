# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    """

    def __init__(self, loss: dict, temperature: float = 0.1) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)
        self.temperature = temperature

    def forward(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Forward function to compute contrastive loss.

        Args:
            pos (torch.Tensor): Nx1 positive similarity.
            neg (torch.Tensor): Nxk negative similarity.

        Returns:
            torch.Tensor: The contrastive loss.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)

        loss = self.loss(logits, labels)
        return loss
