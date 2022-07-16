# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.data import SelfSupDataSample
from ..builder import MODELS
from .base import BaseModel


@MODELS.register_module()
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.

        Returns:
            torch.Tensor: Outputs from neck.
        """
        latent, _, ids_restore = self.backbone(batch_inputs[0])
        pred = self.neck(latent, ids_restore)
        return pred

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
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        latent, mask, ids_restore = self.backbone(batch_inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, batch_inputs[0], mask)
        losses = dict(loss=loss)
        return losses
