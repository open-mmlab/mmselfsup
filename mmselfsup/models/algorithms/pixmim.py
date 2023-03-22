# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .mae import MAE


@MODELS.register_module()
class PixMIM(MAE):
    """The official implementation of PixMIM.

    Implementation of `PixMIM: Rethinking Pixel Reconstruction in
    Masked Image Modeling <https://arxiv.org/pdf/2303.02416.pdf>`_.

    Please refer to MAE for these initialization arguments.
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
        low_freq_targets = self.target_generator(inputs[0])
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, low_freq_targets, mask)
        losses = dict(loss=loss)
        return losses
