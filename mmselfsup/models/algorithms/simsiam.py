# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.core import SelfSupDataSample
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class SimSiam(BaseModel):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_. The operation of fixing learning rate
    of predictor is in `core/hooks/simsiam_hook.py`.
    """

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        return self.backbone(batch_inputs[0])

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
        img_v1 = batch_inputs[0]
        img_v2 = batch_inputs[1]

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        loss_1 = self.head(z1, z2)
        loss_2 = self.head(z2, z1)

        losses = dict(loss=0.5 * (loss_1 + loss_2))
        return losses
