# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseModel
from .beit import BEiT
from .byol import BYOL
from .cae import CAE
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .interclr_moco import InterCLRMoCo
from .mae import MAE
from .maskfeat import MaskFeat
<<<<<<< HEAD
from .milan import MILAN
=======
from .mmcls_classifier_wrapper import MMClsImageClassifierWrapper
>>>>>>> upstream/master
from .moco import MoCo
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simmim import SimMIM
from .simsiam import SimSiam
from .swav import SwAV
from .dino import DINO

__all__ = [
<<<<<<< HEAD
    'BaseModel', 'BarlowTwins', 'BEiT', 'BYOL', 'DeepCluster', 'DenseCL',
    'MoCo', 'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR', 'SimSiam',
    'SwAV', 'MAE', 'MoCoV3', 'SimMIM', 'CAE', 'MaskFeat', 'MILAN', 'DINO'
=======
    'BaseModel', 'BarlowTwins', 'BYOL', 'Classification', 'DeepCluster',
    'DenseCL', 'InterCLRMoCo', 'MoCo', 'NPID', 'ODC', 'RelativeLoc',
    'RotationPred', 'SimCLR', 'SimSiam', 'SwAV', 'MAE', 'MoCoV3', 'SimMIM',
    'MMClsImageClassifierWrapper', 'CAE', 'MaskFeat'
>>>>>>> upstream/master
]
