# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.data import SelfSupDataSample
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class BarlowTwins(BaseModel):
    """BarlowTwins.

    Implementation of `Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction <https://arxiv.org/abs/2103.03230>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/barlowtwins/blob/main/main.py>`_.
    """

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(batch_inputs[0])
        return x

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(batch_inputs, list)
        img_v1 = batch_inputs[0]
        img_v2 = batch_inputs[1]

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        loss = self.head(z1, z2)
        losses = dict(loss=loss)
        return losses
