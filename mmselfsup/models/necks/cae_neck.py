# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_

from mmselfsup.registry import MODELS
from ..utils import CAETransformerRegressorLayer, TransformerEncoderLayer


@MODELS.register_module()
class CAENeck(BaseModule):
    """Neck for CAE Pre-training.

    This module construct the latent prediction regressor and the decoder
    for the latent prediction and final prediction.

    Args:
        patch_size (int): The patch size of each token. Defaults to 16.
        num_classes (int): The number of classes for final prediction. Defaults
            to 8192.
        embed_dims (int): The embed dims of latent feature in regressor and
            decoder. Defaults to 768.
        regressor_depth (int): The number of regressor blocks. Defaults to 6.
        decoder_depth (int): The number of decoder blocks. Defaults to 8.
        num_heads (int): The number of head in multi-head attention. Defaults
            to 12.
        mlp_ratio (int): The expand ratio of latent features in MLP. defaults
            to 4.
        qkv_bias (bool): Whether or not to use qkv bias. Defaults to True.
        qk_scale (float, optional): The scale applied to the results of qk.
            Defaults to None.
        drop_rate (float): The dropout rate. Defaults to 0.
        attn_drop_rate (float): The dropout rate in attention block. Defaults
            to 0.
        norm_cfg (dict): The config of normalization layer. Defaults to
            dict(type='LN', eps=1e-6).
        init_values (float, optional): The init value of gamma. Defaults to
            None.
        mask_tokens_num (int): The number of mask tokens. Defaults to 75.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 patch_size: int = 16,
                 num_classes: int = 8192,
                 embed_dims: int = 768,
                 regressor_depth: int = 6,
                 decoder_depth: int = 8,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_values: float = None,
                 mask_tokens_num: int = 75,
                 init_cfg: dict = None) -> None:
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
            CAETransformerRegressorLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=regressor_drop_path_rates[i],
                norm_cfg=norm_cfg,
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

        _, self.norm_regressor = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm_decoder = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)

        self.head = nn.Linear(
            embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

    def init_weights(self) -> None:
        """Initialization."""
        super().init_weights()
        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.head.weight, std=0.02)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
            self, x_unmasked: torch.Tensor, pos_embed_masked: torch.Tensor,
            pos_embed_unmasked: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the latent prediction and final prediction.

        Args:
            x_unmasked (torch.Tensor): Features of unmasked tokens.
            pos_embed_masked (torch.Tensor): Position embedding of masked
                tokens.
            pos_embed_unmasked (torch.Tensor): Position embedding of unmasked
                tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Final prediction and latent
                prediction.
        """
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
