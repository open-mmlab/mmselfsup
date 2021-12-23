# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .mocov3_head import MoCoV3Head
from .mae_cls_head import VitHead
from .multi_cls_head import MultiClsHead
from .pixel_head import PixelHead
from .swav_head import SwAVHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'MultiClsHead', 'SwAVHead', 'PixelHead', 'VitHead', 'MoCoV3Head'
]
