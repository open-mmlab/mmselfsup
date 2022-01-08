# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .mocov3_head import MoCoV3Head
from .soft_cls_head import SoftClsHead
from .multi_cls_head import MultiClsHead
from .mse_pretrain_head import MAEPretrainHead
from .swav_head import SwAVHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'MultiClsHead', 'SwAVHead', 'SoftClsHead', 'MAEPretrainHead', 'MoCoV3Head'
]
