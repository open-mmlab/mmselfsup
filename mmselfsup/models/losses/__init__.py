# Copyright (c) OpenMMLab. All rights reserved.
from .beit_loss import BEiTLoss
from .cae_loss import CAELoss
from .cosine_similarity_loss import CosineSimilarityLoss
from .cross_correlation_loss import CrossCorrelationLoss
from .mae_loss import MAEReconstructionLoss
from .milan_loss import MILANReconstructionLoss
from .reconstruction_loss import PixelReconstructionLoss
from .simmim_loss import SimMIMReconstructionLoss
from .swav_loss import SwAVLoss

__all__ = [
    'BEiTLoss', 'CAELoss', 'CrossCorrelationLoss', 'CosineSimilarityLoss',
    'MAEReconstructionLoss', 'SimMIMReconstructionLoss', 'SwAVLoss',
    'PixelReconstructionLoss', 'MILANReconstructionLoss'
]
