# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch

from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils.hog_layer import HOGLayerC
from .base import BaseModel


@ALGORITHMS.register_module()
class MaskFeat(BaseModel):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for
    Self-Supervised Visual Pre-Training <https://arxiv.org/abs/2112.09133>`_.
    Args:
        backbone (dict): Config dict for encoder.
        head (dict): Config dict for loss functions.
        hog_para (dict): Config dict for hog layer.
            dict['nbins', int]: Number of bin. Defaults to 9.
            dict['pool', float]: Number of cell. Defaults to 8.
            dict['gaussian_window', int]: Size of gaussian kernel.
                Defaults to 16.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 head: dict,
                 hog_para: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)
        assert hog_para is not None
        self.hog_layer = HOGLayerC(**hog_para)

    def extract_feat(self, input: List[torch.Tensor]) -> torch.Tensor:
        """Function to extract features from backbone.

        Args:
            input (List[torch.Tensor, torch.Tensor]): Input images and masks.
        Returns:
            tuple[Tensor]: backbone outputs.
        """
        img = input[0]
        mask = input[1]
        return self.backbone(img, mask)

    def forward_train(self, input: List[torch.Tensor], **kwargs) -> dict:
        """Forward computation during training.

        Args:
            input (List[torch.Tensor, torch.Tensor]): Input images and masks.
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img = input[0]
        mask = input[1]

        hog = self.hog_layer(img)
        latent = self.backbone(img, mask)
        losses = self.head(latent, hog, mask)

        return losses
