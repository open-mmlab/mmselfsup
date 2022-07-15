# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule
from torch import nn

from mmselfsup.registry import MODELS


@MODELS.register_module()
class CosineSimilarityLoss(BaseModule):
    """Cosine similarity loss function.

    Compute the similarity between two features and optimize that similarity as
    loss.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Forward function of cosine similarity loss.

        Args:
            pred (torch.Tensor): The predicted features.
            target (torch.Tensor): The target features.

        Returns:
            torch.Tensor: The cosine similarity loss.
        """
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -(pred_norm * target_norm).sum(dim=1).mean()
        return loss
