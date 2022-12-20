# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class MixMIM(BaseModel):
    """MiXMIM.

    Implementation of `MixMIM: Mixed and Masked Image Modeling for Efficient
    Visual Representation Learning. <https://arxiv.org/abs/2205.13137>`_.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):

        head.update(dict(patch_size=neck.encoder_stride))
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

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
        latent, mask = self.backbone(inputs[0])
        x_rec = self.neck(latent, mask)
        loss = self.head(x_rec, inputs[0], mask)
        losses = dict(loss=loss)
        return losses
