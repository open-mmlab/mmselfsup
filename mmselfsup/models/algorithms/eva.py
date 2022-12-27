# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class EVA(BaseModel):
    """EVA.

    Implementation of `EVA: Exploring the Limits of Masked Visual
    Representation Learning at Scale <https://arxiv.org/abs/2211.07636>`_.
    """

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

        clip_feature, _ = self.target_generator(inputs[0])

        latent, mask, ids_restore = self.backbone(inputs[0])

        pred = self.neck(latent, ids_restore)

        clip_feature = clip_feature[:, 1:, :]
        loss = self.head(pred, clip_feature, mask)

        losses = dict(loss=loss)
        return losses
