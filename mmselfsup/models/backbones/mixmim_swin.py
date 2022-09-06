# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmcls.models import SwinTransformer
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmselfsup.registry import MODELS

import random
from torch.nn import functional as F
from torch import nn
from mmcls.models.utils.attention import WindowMSA
from mmcv.runner.base_module import BaseModule
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed, PatchMerging
from mmcv.cnn import build_norm_layer
from ..utils import build_2d_sincos_position_embedding

class MixMIMBlock(BaseModule):

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(B, 1, 1)   # B, N, 1
            attn_mask = attn_mask.view(B, H, W, 1)
            attn_mask = window_partition(attn_mask, self.window_size)
            attn_mask = attn_mask.view(-1, self.window_size * self.window_size, 1)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class MixMIMLayer(BaseModule):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                MixMIMBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer)
            )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, attn_mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask=attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

@MODELS.register_module()
class MixMIMTransformer(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
        **dict.fromkeys(['h', 'huge'],
                        {'embed_dims': 352,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [11, 22, 44, 88]}),
    }  # yapf: disable



    def __init__(self,
                 arch='base',
                 mlp_ratio=4,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dims=96,
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 patch_cfg=dict(),
                 norm_cfg=dict(type='LN'),
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 init_cfg: Optional[dict] = None,
                 ) -> None:
        super(MixMIMTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']

        self.encoder_stride = 32

        self.num_layers = len(self.depths)
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(MixMIMLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(self.patch_resolution[0] // (2 ** i_layer), self.patch_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=self.use_checkpoint)
            )

        self.num_features = int(self.embed_dims * 2 ** (self.num_layers - 1))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm = build_norm_layer(norm_cfg, self.num_features)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))

        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims), requires_grad=False)

    def init_weights(self):
        super(MixMIMTransformer, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        trunc_normal_(self.mask_token, mean=0., std=.02)
        trunc_normal_(self.absolute_pos_embed, std=0.02)

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5), self.absolute_pos_embed.shape[-1], cls_token=False)
        self.absolute_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x: torch.Tensor, mask_ratio: float = 0.5):

        B, C, H, W = x.shape
        out_H = H // self.encoder_stride
        out_W = W // self.encoder_stride
        s3_H, s3_W = out_H * 2, out_W * 2
        s2_H, s2_W = out_H * 4, out_W * 4
        s1_H, s1_W = out_H * 8, out_W * 8

        seq_l = out_H * out_W
        # use a shared mask for a batch images
        mask = torch.zeros([1, 1, seq_l], device=x.device)

        mask_ratio = mask_ratio + random.uniform(0.0, self.range_mask_ratio)
        noise = torch.rand(1, 1, seq_l, device=x.device)  # noise in [0, 1]
        # ascend: small is keep, large is remove
        mask_idx = torch.argsort(noise, dim=2)[:, :, :int(seq_l * mask_ratio)]
        mask.scatter_(2, mask_idx, 1)
        mask = mask.reshape(1, 1, out_H, out_W)
        mask_s1 = F.interpolate(mask, size=(s1_H, s1_W), mode='nearest')
        mask_s2 = F.interpolate(mask, size=(s2_H, s2_W), mode='nearest')
        mask_s3 = F.interpolate(mask, size=(s3_H, s3_W), mode='nearest')

        mask = mask.reshape(1, out_H * out_W, 1).contiguous()
        mask_s1 = mask_s1.reshape(1, s1_H * s1_W, 1).contiguous()
        mask_s2 = mask_s2.reshape(1, s2_H * s2_W, 1).contiguous()
        mask_s3 = mask_s3.reshape(1, s3_H * s3_W, 1).contiguous()

        return mask_s1, mask_s2, mask_s3, mask


    def forward(self, x: torch.Tensor, mask_ratio=0.5):

        mask_s1, mask_s2, mask_s3, mask_s4 = self.random_masking(x, mask_ratio)

        x = self.patch_embed(x)

        B, L, _ = x.shape
        H = W = int(L ** 0.5)

        x = x * (1. - mask_s1) + x.flip(0) * mask_s1
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                x = layer(x, attn_mask=mask_s1)
            elif idx == 1:
                x = layer(x, attn_mask=mask_s2)
            elif idx == 2:
                x = layer(x, attn_mask=mask_s3)
            elif idx == 3:
                x = layer(x, attn_mask=mask_s4)
        x = self.norm(x)

        return x

