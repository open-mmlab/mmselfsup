# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class BEiT(BaseModel):
    """BEiT v1/v2.

    Implementation of `BEiT: BERT Pre-Training of Image Transformers
    <https://arxiv.org/abs/2106.08254>`_ and `BEiT v2: Masked Image Modeling
    with Vector-Quantized Visual Tokenizers
    <https://arxiv.org/abs/2208.06366>`_.
    """

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])

        img_latent = self.backbone(batch_inputs[0], mask)

        # batch_inputs[1] is the target image
        with torch.no_grad():
            target = self.target_generator(batch_inputs[1])
            target = target.detach()

        if self.with_neck:
            # BEiT v2
            feats, feats_cls_pt = self.neck(
                img_latent, rel_pos_bias=self.backbone.shared_rel_pos_bias)
            loss = self.head(feats, feats_cls_pt, target, mask)
        else:
            # BEiT v1
            loss = self.head(img_latent[0], target, mask)

        if isinstance(loss, torch.Tensor):
            losses = dict(loss=loss)
            return losses
        elif isinstance(loss, Tuple):
            # the loss_1 and loss_2 are general reconstruction loss (patch
            # feature vectors from last layer of backbone) and early state
            # reconstruction loss (patch feature vectors from intermediate
            # layer of backbone)
            loss_1, loss_2 = loss[0], loss[1]
            losses = dict()
            # the key with prefix 'loss', like loss_1 and loss_2, will be used
            # as the final criterion
            losses['loss_1'] = loss_1
            losses['loss_2'] = loss_2
            return losses
