from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from timm.models.vision_transformer import Block, PatchEmbed

from ..builder import BACKBONES


@BACKBONES.register_module()
class MAEFinetuneViT(BaseModule):
    """Vision Transformer for MAE classification benchmark."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=None,
                 init_cfg=None,
                 global_pool=False,
                 finetune=True):
        super(MAEFinetuneViT, self).__init__(init_cfg)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        self.finetune = finetune
        if not self.finetune:
            self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def train(self, mode=True):
        super(MAEFinetuneViT, self).train(mode)
        if not self.finetune:
            self._freeze_stages()

    def _freeze_stages(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            m = getattr(self, name.split('.')[0])
            if isinstance(m, nn.Module):
                m.eval()
