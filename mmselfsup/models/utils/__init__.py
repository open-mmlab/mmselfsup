# Copyright (c) OpenMMLab. All rights reserved.
from .dall_e import Encoder
from .data_preprocessor import (CAEDataPreprocessor,
                                RelativeLocDataPreprocessor,
                                RotationPredDataPreprocessor,
                                SelfSupDataPreprocessor,
                                TwoNormDataPreprocessor)
from .ema import CosineEMA
from .extractor import Extractor
from .gather_layer import GatherLayer
from .multi_pooling import MultiPooling
from .multi_prototypes import MultiPrototypes
from .norm_ema_quantizer import NormEMAVectorQuantizer
from .position_embedding import build_2d_sincos_position_embedding
from .sobel import Sobel
from .transformer_blocks import (BEiTV2ClsPretrainLayers,
                                 CAETransformerRegressorLayer,
                                 MultiheadAttention, RelativePositionBias,
                                 TransformerEncoderLayer)

try:
    from .res_layer_extra_norm import ResLayerExtraNorm
except ImportError:
    ResLayerExtraNorm = None

__all__ = [
    'Extractor', 'GatherLayer', 'MultiPooling', 'MultiPrototypes',
    'build_2d_sincos_position_embedding', 'Sobel', 'MultiheadAttention',
    'TransformerEncoderLayer', 'CAETransformerRegressorLayer', 'Encoder',
    'CosineEMA', 'SelfSupDataPreprocessor', 'RelativeLocDataPreprocessor',
    'RotationPredDataPreprocessor', 'CAEDataPreprocessor', 'ResLayerExtraNorm',
    'RelativePositionBias', 'BEiTV2ClsPretrainLayers',
    'NormEMAVectorQuantizer', 'TwoNormDataPreprocessor'
]
