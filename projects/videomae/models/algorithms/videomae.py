# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.models.algorithms import BaseModel
from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample


@MODELS.register_module()
class VideoMAE(BaseModel):
    """VideoMAE algorithm.

    Implementation of  `VideoMAE: Masked Autoencoders are Data-Efficient
    Learners for Self-Supervised Video Pre-Training
    <https://arxiv.org/pdf/2203.12602.pdf>`_
    """

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """"""
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        video_latent = self.backbone(inputs[0], mask)
        feat = self.neck(video_latent[0])
        return feat

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training."""
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        video = inputs[0].squeeze(1)
        # change the mask from the float to bool type
        mask = mask.to(torch.bool)
        # encoder part
        video = self.backbone(video, mask)
        # decoder part
        video_rec = self.neck(video, mask)
        # criterion part
        loss = self.head(video_rec, inputs[0].squeeze(1), mask)
        losses = dict(loss=loss)
        return losses
