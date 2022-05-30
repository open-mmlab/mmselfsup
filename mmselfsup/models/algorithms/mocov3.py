# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.core import SelfSupDataSample
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_neck)
from ..utils import CosineEMA
from .base import BaseModel


@ALGORITHMS.register_module()
class MoCoV3(BaseModel):
    """MoCo v3.

    Implementation of `An Empirical Study of Training Self-Supervised Vision
    Transformers <https://arxiv.org/abs/2104.02057>`_.

    Args:
        backbone (Dict): Config dict for module of backbone
        neck (Dict): Config dict for module of deep features to compact feature
            vectors.
        head (Dict): Config dict for module of head functions.
        loss (Dict): Config dict for module of loss functions.
        base_momentum (float, , optional): Momentum coefficient for the
            momentum-updated encoder. Defaults to 0.99.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Dict or list[Dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 backbone: Dict,
                 neck: Dict,
                 head: Dict,
                 loss: Dict,
                 base_momentum: Optional[float] = 0.99,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[List[Dict], Dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        assert neck is not None
        self.base_encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.base_encoder[0]
        self.neck = self.base_encoder[1]
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)

        # create momentum model
        self.momentum_encoder = CosineEMA(
            self.base_encoder, momentum=base_momentum)
        for param_m in self.momentum_encoder.module.parameters():
            param_m.requires_grad = False

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x

    def forward_train(self, inputs: List[torch.Tensor],
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
        view_1 = inputs[0]
        view_2 = inputs[1]

        # compute query features, [N, C] each
        q1 = self.base_encoder(view_1)[0]
        q2 = self.base_encoder(view_2)[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(self.base_encoder)

            k1 = self.momentum_encoder(view_1)[0]
            k2 = self.momentum_encoder(view_2)[0]

        logits_1, labels_1 = self.head(q1, k2)
        logits_2, labels_2 = self.head(q2, k1)

        loss_1 = self.loss(logits_1, labels_1)
        loss_2 = self.loss(logits_2, labels_2)

        losses = dict(loss=loss_1 + loss_2)
        return losses
