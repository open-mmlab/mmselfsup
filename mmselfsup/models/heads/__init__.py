# Copyright (c) OpenMMLab. All rights reserved.
from .cae_head import CAEHead
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .mae_head import MAEFinetuneHead, MAELinprobeHead, MAEPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .simmim_head import SimMIMHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead',
    'LatentCrossCorrelationHead', 'MultiClsHead', 'MAEFinetuneHead',
    'MAEPretrainHead', 'MoCoV3Head', 'SimMIMHead', 'CAEHead', 'MAELinprobeHead'
]
