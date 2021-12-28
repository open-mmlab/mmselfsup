# Copyright (c) OpenMMLab. All rights reserved.
import torch
from timm.models.layers import trunc_normal_
from torch import nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils.mae_blocks import get_sinusoid_encoding_table
from .base import BaseModel


@ALGORITHMS.register_module('MAE')
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
     <https://arxiv.org/abs/2111.06377>`_.
    Part of the code is borrowed from:
    `<https://github.com/pengzhiliang/MAE-pytorch>`_.

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
        assert head is not None
        self.head = build_head(head)

        self.encoder_to_decoder = nn.Linear(
            backbone['embed_dim'], neck['embed_dim'], bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, neck['embed_dim']))

        self.pos_embed = get_sinusoid_encoding_table(
            self.backbone.patch_embed.num_patches, neck['embed_dim'])

        trunc_normal_(self.mask_token, std=.02)

    def extract_feat(self, img, mask):

        return self.backbone(img, mask)

    def forward_train(self, x, mask, target):
        x_vis = self.backbone(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, _, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(
            x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.neck(x_full,
                      pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]
        losses = self.head(x, target)

        return losses
