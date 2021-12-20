from functools import partial

import torch
import torch.nn as nn

from ..builder import NECKS
from ..utils import Block


@NECKS.register_module('MaeNeck')
class PretrainVisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
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
        init_values=None,
        num_patches=196,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * patch_size**2
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
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
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])
                          )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]

        return x
