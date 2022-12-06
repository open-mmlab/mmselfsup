# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
from einops import rearrange
from mmcls.models import BEiT
from mmengine.model import BaseModule
from torch import nn

from mmselfsup.models.utils import NormEMAVectorQuantizer
from mmselfsup.registry import MODELS


@MODELS.register_module()
class VQKD(BaseModule):
    """Vector-Quantized Knowledge Distillation.

    The module only contains encoder and VectorQuantizer part
    Modified from https://github.com/microsoft/unilm/blob/master/beit2/modeling_vqkd.py

    Args:
        encoder_config (dict): The config of encoder.
        decoder_config (dict, optional): The config of decoder. Currently,
            VQKD only support to build encoder. Defaults to None.
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        beta (float): The mutiplier for VectorQuantizer loss. Defaults to 1.
        quantize_kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 encoder_config: dict,
                 decoder_config: Optional[dict] = None,
                 num_embed: int = 8192,
                 embed_dims: int = 32,
                 decay: float = 0.99,
                 beta: float = 1.0,
                 quantize_kmeans_init: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.encoder = BEiT(**encoder_config)
        if decoder_config is not None:
            self.decoder = BEiT(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            num_embed=num_embed,
            embed_dims=embed_dims,
            beta=beta,
            decay=decay,
            kmeans_init=quantize_kmeans_init,
        )

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder.arch_settings['embed_dims'],
                      self.encoder.arch_settings['embed_dims']), nn.Tanh(),
            nn.Linear(self.encoder.arch_settings['embed_dims'], embed_dims))

    def get_tokens(self, x: torch.Tensor) -> dict:
        """Get tokens for beit pre-training."""
        _, embed_ind, _ = self.encode(x)
        output = {}
        output['token'] = embed_ind.view(x.shape[0], -1)
        output['input_img'] = x

        return output

    def encode(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input images and get corresponding results."""
        encoder_features = self.encoder(x)[0]
        B, C, N1, N2 = encoder_features.shape
        encoder_features = encoder_features.permute(0, 2, 3,
                                                    1).reshape(B, N1 * N2, C)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function.

        Currently, only support to get tokens.
        """
        return self.get_tokens(x)['token']
