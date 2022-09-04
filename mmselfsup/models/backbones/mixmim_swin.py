# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmcls.models import SwinTransformer
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmselfsup.registry import MODELS
from ..utils import build_2d_sincos_position_embedding
import random
from torch.nn import functional as F
from torch import nn
from mmcls.models.utils.attention import WindowMSA
from mmcv.runner.base_module import BaseModule
from torch.utils.checkpoint import checkpoint


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
    def __init__(self,
                 decoder_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4,
                 norm_pix_loss=True,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=96,
                 depths=[2, 2, 18, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7, 14, 7],
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 range_mask_ratio=0.0,
                 init_cfg: Optional[dict] = None,
                 ) -> None:
        super(MixMIMTransformer, self).__init__(init_cfg=init_cfg)
        pass



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

