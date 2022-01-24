# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .mae_pretrain_head import MAEPretrainHead
from .mae_finetune_head import MAEFinetuneHead
from .mae_linprobe_head import MAELinprobeHead
from .swav_head import SwAVHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'MultiClsHead', 'SwAVHead', 'MAEFinetuneHead', 'MAEPretrainHead',
    'MAELinprobeHead', 'MoCoV3Head'
]
