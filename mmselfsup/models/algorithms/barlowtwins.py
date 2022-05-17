# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

from mmselfsup.core import SelfSupDataSample
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class BarlowTwins(BaseModel):
    """BarlowTwins.

    Implementation of `Barlow Twins: Self-Supervised Learning via Redundancy
    Reduction <https://arxiv.org/abs/2103.03230>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/barlowtwins/blob/main/main.py>`_.

    Args:
        backbone (Dict, optional): Config dict for module of backbone.
            Defaults to None.
        neck (Dict, optional): Config dict for module of deep features
            to compact feature vectors. Defaults to None.
        head (Dict, optional): Config dict for module of loss functions.
            Defaults to None.
        preprocess_cfg (Dict, optional): Config dict to preprocess images.
            Defaults to None.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

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

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        loss_dict = self.head(z1, z2)
        return loss_dict
