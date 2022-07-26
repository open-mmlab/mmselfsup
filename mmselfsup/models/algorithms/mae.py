# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class MAE(BaseModel):
    """MAE. Implementation of `Masked Autoencoders Are Scalable Vision
    Learners.

     <https://arxiv.org/abs/2111.06377>`_.
    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict = None,
                 neck: dict = None,
                 head: dict = None,
                 init_cfg: dict = None) -> None:
        super(MAE, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super(MAE, self).init_weights()

    def extract_feat(self, img: torch.Tensor) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
        Returns:
            tuple[torch.Tensor]: backbone outputs.
        """
        return self.backbone(img)

    def forward_train(self, img: torch.Tensor,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        latent, mask, ids_restore = self.backbone(img)
        pred = self.neck(latent, ids_restore)
        losses = self.head(img, pred, mask)

        return losses

    def forward_test(self, img: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward computation during testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        latent, mask, ids_restore = self.backbone(img)
        pred = self.neck(latent, ids_restore)

        pred = self.head.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred
