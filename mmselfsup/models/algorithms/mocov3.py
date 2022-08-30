# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
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
        base_momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.99.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.99,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.momentum_encoder = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum)

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x

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
        view_1 = inputs[0]
        view_2 = inputs[1]

        # compute query features, [N, C] each
        q1 = self.neck(self.backbone(view_1))[0]
        q2 = self.neck(self.backbone(view_2))[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # update momentum encoder
            self.momentum_encoder.update_parameters(
                nn.Sequential(self.backbone, self.neck))

            k1 = self.momentum_encoder(view_1)[0]
            k2 = self.momentum_encoder(view_2)[0]

        loss = self.head(q1, k2) + self.head(q2, k1)

        losses = dict(loss=loss)
        return losses
