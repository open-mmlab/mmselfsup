# Copyright (c) OpenMMLab. All rights reserved.
from .beitv1_head import BEiTV1Head
from .beitv2_head import BEiTV2Head
from .cae_head import CAEHead
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .mae_head import MAEPretrainHead
from .maskfeat_head import MaskFeatPretrainHead
from .milan_head import MILANPretrainHead
from .mixmim_head import MixMIMPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .simmim_head import SimMIMHead
from .swav_head import SwAVHead

__all__ = [
    'BEiTV1Head', 'BEiTV2Head', 'ContrastiveHead', 'ClsHead',
    'LatentPredictHead', 'LatentCrossCorrelationHead', 'MultiClsHead',
    'MAEPretrainHead', 'MoCoV3Head', 'SimMIMHead', 'CAEHead', 'SwAVHead',
    'MaskFeatPretrainHead', 'MILANPretrainHead', 'MixMIMPretrainHead'
]
