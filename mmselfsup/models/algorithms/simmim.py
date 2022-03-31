# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class SimMIM(BaseModel):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super(SimMIM, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, img: torch.Tensor) -> tuple:
        """Function to extract features from backbone.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Latent representations of images.
        """
        return self.backbone(img)

    def forward_train(self, x: List[torch.Tensor], **kwargs) -> dict:
        """Forward the masked image and get the reconstruction loss.

        Args:
            x (List[torch.Tensor, torch.Tensor]): Images and masks.

        Returns:
            dict: Reconstructed loss.
        """
        img, mask = x

        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent[0])
        losses = self.head(img, img_rec, mask)

        return losses
