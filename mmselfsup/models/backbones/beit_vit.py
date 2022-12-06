# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Tuple, Union

import torch
from mmcls.models import BEiT, resize_pos_embed
from mmengine.model.weight_init import trunc_normal_
from torch import nn

from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTViT(BEiT):
    """Vision Transformer for BEiT pre-training.

    Rewritten version of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        avg_token (bool): Whether or not to use the mean patch token for
            classification. If True, the model will only take the average
            of all patch tokens. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        use_abs_pos_emb (bool): Whether or not use absolute position embedding.
            Defaults to False.
        use_rel_pos_bias (bool): Whether or not use relative position bias.
            Defaults to False.
        use_shared_rel_pos_bias (bool): Whether or not use shared relative
            position bias. Defaults to True.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN. Defaults to 0.1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_indices: int = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 avg_token: bool = False,
                 frozen_stages: int = -1,
                 output_cls_token: bool = True,
                 use_abs_pos_emb: bool = False,
                 use_rel_pos_bias: bool = False,
                 use_shared_rel_pos_bias: bool = True,
                 layer_scale_init_value: int = 0.1,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(padding=0),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            avg_token=avg_token,
            frozen_stages=frozen_stages,
            output_cls_token=output_cls_token,
            use_abs_pos_emb=use_abs_pos_emb,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            use_rel_pos_bias=use_rel_pos_bias,
            layer_scale_init_value=layer_scale_init_value,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        self.rescale_init_weight()

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data, layer_id + 1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor]:
        """The BEiT style forward function.

        Args:
            x (torch.Tensor): Input images, which is of shape (B x C x H x W).
            mask (torch.Tensor): Mask for input, which is of shape
                (B x patch_resolution[0] x patch_resolution[1]).

        Returns:
            Tuple[torch.Tensor]: Hidden features.
        """
        x, patch_resolution = self.patch_embed(x)

        # replace the masked visual tokens by mask_token
        B, L, _ = x.shape
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1. - w) + mask_token * w

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        self.shared_rel_pos_bias = self.rel_pos_bias().to(
            mask.device) if self.rel_pos_bias is not None else None

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_pos_bias=self.shared_rel_pos_bias)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
