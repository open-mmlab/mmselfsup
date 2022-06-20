# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.core import SelfSupDataSample
from mmselfsup.registry import MODELS
from ..utils import CosineEMA
from .base import BaseModel


@MODELS.register_module()
class MoCoV3(BaseModel):
    """MoCo v3.

    Implementation of `An Empirical Study of Training Self-Supervised Vision
    Transformers <https://arxiv.org/abs/2104.02057>`_.

    Args:
        backbone (dict): Config dict for module of backbone
        neck (dict): Config dict for module of deep features to compact feature
            vectors.
        head (dict): Config dict for module of head functions.
        base_momentum (float, , optional): Momentum coefficient for the
            momentum-updated encoder. Defaults to 0.99.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Dict or list[Dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: Optional[float] = 0.99,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.base_encoder = nn.Sequential(self.backbone, self.neck)

        # create momentum model
        self.momentum_encoder = CosineEMA(
            self.base_encoder, momentum=base_momentum)
        for param_m in self.momentum_encoder.module.parameters():
            param_m.requires_grad = False

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
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
        """The forward function in training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        view_1 = batch_inputs[0]
        view_2 = batch_inputs[1]

        # compute query features, [N, C] each
        q1 = self.base_encoder(view_1)[0]
        q2 = self.base_encoder(view_2)[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(self.base_encoder)

            k1 = self.momentum_encoder(view_1)[0]
            k2 = self.momentum_encoder(view_2)[0]

        loss = self.head(q1, k2) + self.head(q2, k1)

        losses = dict(loss=loss)
        return losses
