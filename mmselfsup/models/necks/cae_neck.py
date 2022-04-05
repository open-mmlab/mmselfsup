# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import trunc_normal_
from ..utils import CAETransformerDecoderLayer, TransformerEncoderLayer
from functools import partial

from ..builder import NECKS


@NECKS.register_module()
class CAENeck(BaseModule):

    def __init__(self,
                 patch_size=16,
                 num_classes=8192,
                 embed_dims=768,
                 regressor_depth=6,
                 decoder_depth=8,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=None,
                 mask_tokens_num=75,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_features = self.embed_dim = embed_dims
        self.patch_size = patch_size
        self.mask_token_num = mask_tokens_num

        # regressor
        regressor_drop_path_rates = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, regressor_depth)
        ]
        self.regressors = nn.ModuleList([
            CAETransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=regressor_drop_path_rates[i],
                norm_layer=norm_layer,
                init_values=init_values) for i in range(regressor_depth)
        ])

        # decoder
        decoder_drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)
        ]

        self.decoders = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=decoder_drop_path_rates[i],
                norm_cfg=norm_cfg,
                init_values=init_values) for i in range(decoder_depth)
        ])

        self.norm_regressor = norm_layer(embed_dims)
        self.norm_decoder = norm_layer(embed_dims)

        self.head = nn.Linear(
            embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

    def init_weights(self):
        super(CAENeck, self).init_weights()
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_unmasked, pos_embed_masked, pos_embed_unmasked):
        x_masked = self.mask_token.expand(x_unmasked.shape[0],
                                          self.mask_token_num, -1)
        # regressor
        for regressor in self.regressors:
            x_masked = regressor(
                x_masked, torch.cat([x_unmasked, x_masked], dim=1),
                pos_embed_masked,
                torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1))
        x_masked = self.norm_regressor(x_masked)
        latent_pred = x_masked

        # decoder
        x_masked = x_masked + pos_embed_masked
        for decoder in self.decoders:
            x_masked = decoder(x_masked)
        x_masked = self.norm_decoder(x_masked)

        logits = self.head(x_masked)

        return logits, latent_pred
