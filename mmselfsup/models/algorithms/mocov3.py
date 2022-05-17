# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.core import SelfSupDataSample
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
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
        head (Dict): Config dict for module of loss functions.
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
                 base_momentum: Optional[float] = 0.99,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[List[Dict], Dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        assert neck is not None
        self.base_encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.momentum_encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.base_encoder[0]
        self.neck = self.base_encoder[1]
        assert head is not None
        self.head = build_head(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self) -> None:
        """Initialize base_encoder with init_cfg defined in backbone."""
        super().init_weights()

        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def momentum_update(self) -> None:
        """Momentum update of the momentum encoder."""
        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                1. - self.momentum)

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
            # here we use hook to update momentum encoder, which is a little
            # bit different with the official version but it has negligible
            # influence on the results
            k1 = self.momentum_encoder(view_1)[0]
            k2 = self.momentum_encoder(view_2)[0]

        losses = self.head(q1, k2)['loss'] + self.head(q2, k1)['loss']
        return dict(loss=losses)
