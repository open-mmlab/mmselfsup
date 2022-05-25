# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class ContrastiveHead(BaseModule):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, pos: torch.Tensor,
                neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function to compute contrastive loss.

        Args:
            pos (torch.Tensor): Nx1 positive similarity.
            neg (torch.Tensor): Nxk negative similarity.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the logits
                and labels.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)

        return logits, labels
