# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.core import SelfSupDataSample
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class BYOL(BaseModel):
    """BYOL.

    Implementation of `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_.
    The momentum adjustment is in `core/hooks/byol_hook.py`.

    Args:
        backbone (Dict, optional): Config dict for module of backbone.
        neck (Dict, optional): Config dict for module of deep features
            to compact feature vectors. Defaults to None.
        head (Dict, optional): Config dict for module of head functions.
            Defaults to None.
        loss (Dict, optional): Config dict for module of loss functions.
            Defaults to None.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.996.
        preprocess_cfg (Dict, optional): Config dict to preprocess images.
            Defaults to None.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 loss: Optional[Dict] = None,
                 base_momentum: float = 0.996,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        assert neck is not None
        self.online_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.target_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwargs) -> Tuple[torch.Tensor]:
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
        """Forward computation during training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        img_v1 = inputs[0]
        img_v2 = inputs[1]
        # compute online features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        # compute target features
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0]
            proj_target_v2 = self.target_net(img_v2)[0]

        pred_1, target_1 = self.head(proj_online_v1, proj_target_v2)
        pred_2, target_2 = self.head(proj_online_v2, proj_target_v1)

        loss_1 = self.loss(pred_1, target_1)
        loss_2 = self.loss(pred_2, target_2)

        losses = dict(loss=2. * (loss_1 + loss_2))
        return losses
