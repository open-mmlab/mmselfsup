# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_

from mmselfsup.registry import MODELS
from ..backbones.videomae_vit import VideoMAEBlock


@MODELS.register_module()
class VideoMAEPretrainDecoder(BaseModule):
    """Decoder for VideoMAE Pre-training.

    Some of the code is borrowed from ``. # noqa
    """

    def __init__(self,
                 num_patches: int = 196,
                 patch_size: int = 16,
                 num_classes: int = 768,
                 input_dims: int = 768,
                 embed_dims: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qkv_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_value: Optional[float] = None,
                 tubelet_size: int = 2,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_patches = num_patches

        # used to convert the dim of features from encoder to the dim
        # compatible with that of decoder
        self.decoder_embed = nn.Linear(input_dims, embed_dims, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size**2
        self.embed_dims = embed_dims
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            VideoMAEBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_scale=qkv_scale,
                drop=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                init_values=init_value) for i in range(depth)
        ])

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.head = nn.Linear(embed_dims, num_classes) \
            if num_classes > 0 else nn.Identity()

    def init_weights(self):
        super().init_weights()
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor,
                mask: torch.Tensor, return_token_num: int) -> torch.Tensor:
        """Forward function."""
        B, _, C = x.shape
        x = self.decoder_embed(x)

        expand_pos_embed = pos_embed.expand(x.shape[0], -1, -1)
        visible_pe = expand_pos_embed[~mask].reshape(B, -1, C)
        masked_pe = expand_pos_embed[mask].reshape(B, -1, C)

        x = torch.cat([x + visible_pe, self.mask_token + masked_pe], dim=1)

        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        return x
