# Copyright (c) OpenMMLab. All rights reserved.
from .cae_head import CAEHead
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import (LatentClsHead, LatentCrossCorrelationHead,
                               LatentPredictHead)
from .mae_head import MAEFinetuneHead, MAELinprobeHead, MAEPretrainHead
from .maskfeat_head import MaskFeatFinetuneHead, MaskFeatPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .simmim_head import SimMIMHead
from .swav_head import SwAVHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'LatentCrossCorrelationHead', 'MultiClsHead', 'SwAVHead',
    'MAEFinetuneHead', 'MAEPretrainHead', 'MoCoV3Head', 'SimMIMHead',
    'CAEHead', 'MAELinprobeHead', 'MaskFeatFinetuneHead',
    'MaskFeatPretrainHead'
]
