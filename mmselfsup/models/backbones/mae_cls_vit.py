import torch
from mmcls.models import VisionTransformer
from mmcv.cnn import build_norm_layer
from torch import nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class MAEClsViT(VisionTransformer):
    """Vision Transformer for MAE classification (fine-tuning or linear probe).

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
        global_pool (bool): Whether or not use the global_pooling feature
            for classification. Defaults to True.
        finetune (bool): Whether or not do fine-tuning. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

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
