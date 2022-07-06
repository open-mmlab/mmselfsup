# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.core import SelfSupDataSample
from mmselfsup.registry import MODELS
from ..utils import CosineEMA
from .base import BaseModel


@MODELS.register_module()
class BYOL(BaseModel):
    """BYOL.

    Implementation of `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_.
    The momentum adjustment is in `core/hooks/byol_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features
            to compact feature vectors.
        head (dict): Config dict for module of head functions.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.996.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): Config dict to preprocess images.
            Defaults to None.
        init_cfg (dict or List[dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.996,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.target_net = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum)

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(batch_inputs[0])
        return x

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(batch_inputs, list)
        img_v1 = batch_inputs[0]
        img_v2 = batch_inputs[1]
        # compute online features
        proj_online_v1 = self.neck(self.backbone(img_v1))[0]
        proj_online_v2 = self.neck(self.backbone(img_v2))[0]
        # compute target features
        with torch.no_grad():
            # update the target net
            self.target_net.update_parameters(
                nn.Sequential(self.backbone, self.neck))

            proj_target_v1 = self.target_net(img_v1)[0]
            proj_target_v2 = self.target_net(img_v2)[0]

        loss_1 = self.head(proj_online_v1, proj_target_v2)
        loss_2 = self.head(proj_online_v2, proj_target_v1)

        losses = dict(loss=2. * (loss_1 + loss_2))
        return losses
