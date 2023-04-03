# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseModel
from .beit import BEiT
from .byol import BYOL
from .cae import CAE
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .eva import EVA
from .mae import MAE
from .maskfeat import MaskFeat
from .milan import MILAN
from .mixmim import MixMIM
from .moco import MoCo
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .pixmim import PixMIM
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simmim import SimMIM
from .simsiam import SimSiam
from .swav import SwAV

__all__ = [
    'BaseModel', 'BarlowTwins', 'BEiT', 'BYOL', 'DeepCluster', 'DenseCL',
    'MoCo', 'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR', 'SimSiam',
    'SwAV', 'MAE', 'MoCoV3', 'SimMIM', 'CAE', 'MaskFeat', 'MILAN', 'EVA',
    'MixMIM', 'PixMIM'
]
