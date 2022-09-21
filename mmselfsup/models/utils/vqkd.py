# Copyright (c) OpenMMLab. All rights reserved.
# Copied from https://github.com/microsoft/unilm
# /blob/master/beit2/modeling_vqkd.py

import math

import torch
from einops import rearrange
from mmcls.models import VisionTransformer
from torch import nn

from mmselfsup.models.utils import NormEMAVectorQuantizer


def get_model_default_params():
    return dict(
        arch='base',
        img_size=224,
        patch_size=16,
        in_channels=3,
        out_indices=-1,
        drop_rate=0.,
        drop_path_rate=0.,
        qkv_bias=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        final_norm=True,
        with_cls_token=False,
        avg_token=False,
        frozen_stages=-1,
        output_cls_token=False,
        beit_style=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        layer_scale_init_value=0.,
        interpolate_mode='bicubic',
        patch_cfg=dict(),
        layer_cfgs=dict(),
        init_cfg=None)


class VQKD(nn.Module):

    encoder_config = get_model_default_params()
    n_embed = 8192
    embed_dim = 32
    decay = 0.99
    quantize_kmeans_init = True

    def __init__(self, img_size: int = 224) -> None:
        super().__init__()
        self.encoder_config['img_size'] = img_size

        # encoder params
        self.encoder = VisionTransformer(**self.encoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=self.n_embed,
            embedding_dim=self.embed_dim,
            beta=1.0,
            kmeans_init=self.quantize_kmeans_init,
            decay=self.decay,
        )

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder.arch_settings['embed_dims'],
                      self.encoder.arch_settings['embed_dims']),
            nn.Tanh(),
            nn.Linear(self.encoder.arch_settings['embed_dims'],
                      self.embed_dim)  # for quantize
        )

    def get_tokens(self, data):

        quantize, embed_ind, loss = self.encode(data)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data

        return output

    def encode(self, x):
        encoder_features = self.encoder(x)
        B, C, N1, N2 = encoder_features[0].shape
        encoder_features = encoder_features[0].reshape(B, N1 * N2, C)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(
                encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = int(math.sqrt(N)), int(math.sqrt(N))

        to_quantizer_features = rearrange(
            to_quantizer_features, 'b (h w) c -> b c h w', h=h,
            w=w)  # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss

    def forward(self, x):
        # for beit pre-training
        return self.get_tokens(x)['token']
