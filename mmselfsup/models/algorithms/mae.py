# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

from mmselfsup.core import SelfSupDataSample
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
     <https://arxiv.org/abs/2111.06377>`_.
    Args:
        backbone (Dict, optional): Config dict for encoder. Defaults to None.
        neck (Dict, optional): Config dict for encoder. Defaults to None.
        head (Dict, optional): Config dict for head functions.
            Defaults to None.
        loss (Dict, optional): Config dict for loss functions.
            Defaults to None.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 loss: Optional[Dict] = None,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super(MAE, self).__init__(
            preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)

    def init_weights(self) -> None:
        super(MAE, self).init_weights()

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        return self.backbone(inputs[0])

    def forward_train(self, inputs: List[torch.Tensor],
                      data_samples: List[SelfSupDataSample],
                      **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        target = self.head(inputs[0])
        loss = self.loss(pred, target, mask)
        losses = dict(loss=loss)
        return losses
