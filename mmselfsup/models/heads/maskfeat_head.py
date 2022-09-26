# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MaskFeatPretrainHead(BaseModule):
    """Pre-training head for MaskFeat.

    This head builds a predictor, which can be any registered neck component.
    It also implements construct loss between prediction and target in masked
        region.

    Args:
        predictor (dict): Config dict for module of predictor.
        loss (dict): Config dict for module of loss functions.
    """

    def __init__(self, predictor: dict, loss: dict) -> None:
        super().__init__()
        self.predictor = MODELS.build(predictor)
        self.loss = MODELS.build(loss)

    def init_weights(self):
        nn.init.constant_(self.predictor.fc.bias, 0)
        nn.init.trunc_normal_(self.predictor.fc.weight, std=0.02)

    def forward(self, latent: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward head.

        Args:
            latent (torch.Tensor): Predictions,
                which is of shape B x (1 + L) x C.
            target (torch.Tensor): Hog features, which is of shape B x L x C.
            mask (torch.Tensor): The mask of the hog features,
                which is of shape B x H x W.
        Returns:
            torch.Tensor: The loss tensor.
        """
        B, L, C = latent.shape
        pred = self.predictor([latent.view(B * L, C)])
        pred = pred[0].view(B, L, -1)
        mask = mask.flatten(1).bool()
        loss = self.loss(pred[:, 1:], target, mask)

        return loss
