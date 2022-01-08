# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..utils import get_2d_sincos_pos_embed


@ALGORITHMS.register_module('MAE')
class MAE(BaseModel):
    """MAE.
    backbone (dict): Config dict for encoder.
        Defaults to None.
    neck (dict): Config dict for encoder.
        Defaults to None
    head (dict): Config dict for loss functions.
        Defaults to None.
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
        pos_embed = get_2d_sincos_pos_embed(
            self.backbone.pos_embed.shape[-1],
            int(self.backbone.patch_embed.num_patches**.5),
            cls_token=True)
        self.backbone.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.neck.decoder_pos_embed.shape[-1],
            int(self.backbone.patch_embed.num_patches**.5),
            cls_token=True)
        self.neck.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.backbone.patch_embed.proj.weight.data
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

        return self.backbone(img)

    def forward_train(self, x):

        latent, mask, ids_restore = self.backbone(x)
        pred = self.neck(latent, ids_restore)
        losses = self.head(x, pred, mask)

        return losses
