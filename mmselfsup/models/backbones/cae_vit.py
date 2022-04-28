# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcls.models import VisionTransformer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import ModuleList
from torch import nn

from ..builder import BACKBONES
from ..utils import TransformerEncoderLayer, build_2d_sincos_position_embedding


@BACKBONES.register_module()
class CAEViT(VisionTransformer):
    """Vision Transformer for CAE pre-training.

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
                 arch: str = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: int = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 qkv_bias: bool = True,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 output_cls_token: bool = True,
                 interpolate_mode: str = 'bicubic',
                 init_values: float = None,
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: dict = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)
        self.pos_embed.requires_grad = False
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        # Replace original TransformerEncoderLayer with customized
        # TransformerEncoderLayer
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                init_values=init_values,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

    def init_weights(self) -> None:
        super(CAEViT, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # initialize position  embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())

            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, _ = self.patch_embed(img)
        batch_size, _, dim = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # NOTE: unmasked embeddings
        x_unmasked = x[~mask].reshape(batch_size, -1, dim)
        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

        pos_embed = self.pos_embed.expand(batch_size, self.num_patches + 1,
                                          dim)
        pos_embed_unmasked = pos_embed[:,
                                       1:][~mask].reshape(batch_size, -1, dim)
        pos_embed_unmasked = torch.cat((pos_embed[:, :1], pos_embed_unmasked),
                                       dim=1)
        x_unmasked = x_unmasked + pos_embed_unmasked

        x_unmasked = self.drop_after_pos(x_unmasked)

        for i, layer in enumerate(self.layers):
            x_unmasked = layer(x_unmasked)

            if i == len(self.layers) - 1 and self.final_norm:
                x_unmasked = self.norm1(x_unmasked)

        return x_unmasked
