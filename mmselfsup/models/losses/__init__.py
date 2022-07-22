# Copyright (c) OpenMMLab. All rights reserved.
from .cae_loss import CAELoss
from .cosine_similarity_loss import CosineSimilarityLoss
from .cross_correlation_loss import CrossCorrelationLoss
from .mae_loss import MAEReconstructionLoss
from .reconstruction_loss import PixelReconstructionLoss
from .simmim_loss import SimMIMReconstructionLoss
from .swav_loss import SwAVLoss

__all__ = [
    'CAELoss', 'CrossCorrelationLoss', 'CosineSimilarityLoss',
    'MAEReconstructionLoss', 'SimMIMReconstructionLoss', 'SwAVLoss',
    'PixelReconstructionLoss'
]
