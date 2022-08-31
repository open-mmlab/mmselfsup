# Copyright (c) OpenMMLab. All rights reserved.
from .avgpool2d_neck import AvgPool2dNeck
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .linear_neck import LinearNeck
from .mae_neck import ClsBatchNormNeck, MAEPretrainDecoder
from .mocov2_neck import MoCoV2Neck
from .nonlinear_neck import NonLinearNeck
from .odc_neck import ODCNeck
from .relative_loc_neck import RelativeLocNeck
from .simmim_neck import SimMIMNeck
from .swav_neck import SwAVNeck

__all__ = [
    'AvgPool2dNeck', 'DenseCLNeck', 'LinearNeck', 'MoCoV2Neck',
    'NonLinearNeck', 'ODCNeck', 'RelativeLocNeck', 'SwAVNeck',
    'MAEPretrainDecoder', 'SimMIMNeck', 'CAENeck', 'ClsBatchNormNeck'
]
