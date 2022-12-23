# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class MILAN(BaseModel):
    """MILAN.

    Implementation of `MILAN: Masked Image Pretraining on Language Assisted
    Representation <https://arxiv.org/abs/2208.06049>`_.
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
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        clip_feature, importance = self.target_generator(inputs[0])
        importance = importance[:, 0, 1:]
        latent, ids_restore, ids_keep, ids_dump = self.backbone(
            inputs[0], importance)
        pred = self.neck(latent, ids_restore, ids_keep, ids_dump)
        loss = self.head(pred, clip_feature)
        losses = dict(loss=loss)
        return losses
