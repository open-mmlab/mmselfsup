# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
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

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward function to compute contrastive loss.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses
