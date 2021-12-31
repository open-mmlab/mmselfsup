from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils import Block, PatchEmbed, get_sinusoid_encoding_table


@BACKBONES.register_module()
class Vit(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0.0,
                 init_cfg=None,
                 pretrain=True):
        super().__init__(init_cfg)
        self.embed_dim = embed_dim
        self.pretrain = pretrain

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) if self.pretrain else None
        self.fc_norm = norm_layer(embed_dim) if not self.pretrain else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, mask=None):
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape

        if self.pretrain:
            x = x[~mask].reshape(B, -1, C)
        else:
            x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.pretrain:
            x = self.norm(x)
        else:
            x = self.fc_norm(x.mean(1))

        return x

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        return x
