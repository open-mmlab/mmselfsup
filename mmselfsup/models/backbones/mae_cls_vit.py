import torch
from mmcls.models import VisionTransformer
from mmcv.cnn import build_norm_layer
from torch import nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class MAEClsViT(VisionTransformer):

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 global_pool=True,
                 finetune=True,
                 init_cfg=None):
        super().__init__(arch, img_size, patch_size, out_indices, drop_rate,
                         drop_path_rate, norm_cfg, final_norm,
                         output_cls_token, interpolate_mode, patch_cfg,
                         layer_cfgs, init_cfg)

        self.embed_dims = self.arch_settings['embed_dims']
        self.global_pool = global_pool
        if self.global_pool:
            _, self.fc_norm = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)

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

    def train(self, mode=True):
        super(MAEClsViT, self).train(mode)
        if not self.finetune:
            self._freeze_stages()

    def _freeze_stages(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm1(x)
            outcome = x[:, 0]

        return outcome
