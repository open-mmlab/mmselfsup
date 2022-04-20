# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcls.models import VisionTransformer
from mmcv.cnn import build_norm_layer
from mmcv.runner.base_module import ModuleList

from ..builder import BACKBONES
from ..utils import TransformerEncoderLayer


@BACKBONES.register_module()
class MIMVisionTransformer(VisionTransformer):
    """Vision Transformer for MIM-style model (Mask Image Modeling)
    classification (fine-tuning or linear probe).

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
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
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        finetune (bool): Whether or not do fine-tuning. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 use_window=False,
                 drop_rate=0,
                 drop_path_rate=0,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 init_values=0.0,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 finetune=True,
                 init_cfg=None):
        super().__init__(
            arch,
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
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                window_size=self.patch_resolution if use_window else None,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                init_values=init_values,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        if not self.final_norm:
            _, self.fc_norm = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)

        self.finetune = finetune
        if not self.finetune:
            self._freeze_stages()

    def train(self, mode=True):
        super(MIMVisionTransformer, self).train(mode)
        if not self.finetune:
            self._freeze_stages()

    def _freeze_stages(self):
        """Freeze params in backbone when linear probing."""
        for _, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)[0]

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        if not self.final_norm:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            outcome = x[:, 0]
        return outcome
