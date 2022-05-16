# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

from mmselfsup.core import SelfSupDataSample
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class SimMIM(BaseModel):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.

    Args:
        backbone (Dict): Config dict for encoder. Defaults to None.
        neck (Dict): Config dict for encoder. Defaults to None.
        head (Dict): Config dict for loss functions. Defaults to None.
        preprocess_cfg (Dict): Config to preprocess images.
        init_cfg (Union[List[Dict], Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Dict,
                 neck: Dict,
                 head: Dict,
                 preprocess_cfg: Dict,
                 init_cfg: Optional[Union[List[Dict], Dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

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
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        return self.backbone(inputs[0], mask)

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
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        img = inputs[0]

        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent[0])
        losses = self.head(img, img_rec, mask)

        return losses
