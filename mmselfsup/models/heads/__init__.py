# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .multi_cls_head import MultiClsHead
from .swav_head import SwAVHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'MultiClsHead', 'SwAVHead'
]
