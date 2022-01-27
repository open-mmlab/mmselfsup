# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import build_2d_sincos_position_embedding
from .base import BaseModel


@ALGORITHMS.register_module('MAE')
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
     <https://arxiv.org/abs/2111.06377>`_.
    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self, backbone=None, neck=None, head=None, init_cfg=None):
        super(MAE, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.patch_embed.num_patches
        assert head is not None
        self.head = build_head(head)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = build_2d_sincos_position_embedding(
            int(self.backbone.patch_embed.num_patches**.5),
            self.backbone.pos_embed.shape[-1],
            cls_token=True)
        self.backbone.pos_embed.data.copy_(pos_embed.float())

        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.backbone.patch_embed.num_patches**.5),
            self.neck.decoder_pos_embed.shape[-1],
            cls_token=True)
        self.neck.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        w = self.backbone.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.backbone.cls_token, std=.02)
        torch.nn.init.normal_(self.neck.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        return self.backbone(img)

    def forward_train(self, img, **kwargs):
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
