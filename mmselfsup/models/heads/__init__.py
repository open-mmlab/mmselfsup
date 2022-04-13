# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import (LatentClsHead, LatentCrossCorrelationHead,
                               LatentPredictHead)
from .mae_head import MAEFinetuneHead, MAEPretrainHead, MAELinprobeHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .simmim_head import SimMIMHead
from .swav_head import SwAVHead
from .cae_head import CAEHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'LatentCrossCorrelationHead', 'MultiClsHead', 'SwAVHead',
    'MAEFinetuneHead', 'MAEPretrainHead', 'MoCoV3Head', 'SimMIMHead',
    'CAEHead', 'MAELinprobeHead'
]
