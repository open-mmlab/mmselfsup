# Copyright (c) OpenMMLab. All rights reserved.
from .beit_head import BEiTHead
from .cae_head import CAEHead
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .mae_head import MAEPretrainHead
from .maskfeat_head import MaskFeatPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .simmim_head import SimMIMHead
from .swav_head import SwAVHead

__all__ = [
    'BEiTHead', 'ContrastiveHead', 'ClsHead', 'LatentPredictHead',
    'LatentCrossCorrelationHead', 'MultiClsHead', 'MAEPretrainHead',
    'MoCoV3Head', 'SimMIMHead', 'CAEHead', 'SwAVHead', 'MaskFeatPretrainHead'
]
