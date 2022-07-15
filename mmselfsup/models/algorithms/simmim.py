# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from mmselfsup.data import SelfSupDataSample
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class SimMIM(BaseModel):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.
    """

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> torch.Tensor:
        """The forward function to extract features.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        img_latent = self.backbone(batch_inputs[0], mask)
        feat = self.neck(img_latent[0])
        return feat

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        img = batch_inputs[0]

        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent[0])
        loss = self.head(img_rec, img, mask)
        losses = dict(loss=loss)

        return losses
