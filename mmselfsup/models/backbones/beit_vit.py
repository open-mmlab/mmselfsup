# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import VisionTransformer, resize_pos_embed
from torch import nn

from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTViT(VisionTransformer):
    """Vision Transformer for BEiT pre-training.

    Rewritten version of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        init_values (float, optional): The init value of gamma in
            TransformerEncoderLayer.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: str = 'deit-b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_indices: int = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 qkv_bias: bool = True,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 with_cls_token: bool = True,
                 avg_token: bool = False,
                 frozen_stages: int = -1,
                 output_cls_token: bool = True,
                 beit_style: bool = True,
                 layer_scale_init_value: int = 0.1,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(padding=0),
                 layer_cfgs: dict = dict(),
                 init_cfg: dict = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            with_cls_token=with_cls_token,
            avg_token=avg_token,
            frozen_stages=frozen_stages,
            output_cls_token=output_cls_token,
            beit_style=beit_style,
            layer_scale_init_value=layer_scale_init_value,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(
            1, 1, self.embed_dims))  # torch.Size([1, 1, 768])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # replace the masked visual tokens by mask_token
        L = x.shape[1]
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return x
