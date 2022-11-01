# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class MaskFeat(BaseModel):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2112.09133>`_.
    """

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            inputs (List[torch.Tensor]): The input images and mask.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: Neck outputs.
        """
        img = inputs[0]
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        latent = self.backbone(img, mask)
        return latent

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        img = inputs[0]
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        mask = mask.to(torch.bool)

        latent = self.backbone(img, mask)
        B, L, C = latent.shape
        pred = self.neck([latent.view(B * L, C)])
        pred = pred[0].view(B, L, -1)
        hog = self.target_generator(img)

        loss = self.head(pred, hog, mask)
        losses = dict(loss=loss)
        return losses
