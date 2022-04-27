# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

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
        backbone (dict): Config dict for module of backbone. Defaults to None.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict = None,
                 neck: dict = None,
                 head: dict = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super(BarlowTwins, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, img: torch.Tensor) -> torch.Tensor:
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img: List[torch.Tensor]) -> dict:
        """Forward computation during training.

        Args:
            img (List[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(img, list)
        img_v1 = img[0]
        img_v2 = img[1]

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        losses = self.head(z1, z2)['loss']
        return dict(loss=losses)
