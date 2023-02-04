# Copyright (c) OpenMMLab. All rights reserved.
from .clip import build_clip_model
from .data_preprocessor import (CAEDataPreprocessor,
                                RelativeLocDataPreprocessor,
                                RotationPredDataPreprocessor,
                                SelfSupDataPreprocessor,
                                TwoNormDataPreprocessor,
                                VideoMAEDataPreprocessor)
from .ema import CosineEMA
from .extractor import Extractor
from .gather_layer import GatherLayer
from .multi_pooling import MultiPooling
from .multi_prototypes import MultiPrototypes
from .position_embedding import (build_1d_sincos_position_embedding,
                                 build_2d_sincos_position_embedding)
from .sobel import Sobel
from .transformer_blocks import (CAETransformerRegressorLayer,
                                 MultiheadAttention,
                                 PromptTransformerEncoderLayer,
                                 TransformerEncoderLayer)
from .vector_quantizer import NormEMAVectorQuantizer

try:
    from .res_layer_extra_norm import ResLayerExtraNorm
except ImportError:
    ResLayerExtraNorm = None

__all__ = [
    'Extractor', 'GatherLayer', 'MultiPooling', 'MultiPrototypes',
    'build_2d_sincos_position_embedding', 'Sobel', 'MultiheadAttention',
    'TransformerEncoderLayer', 'CAETransformerRegressorLayer', 'Encoder',
    'CosineEMA', 'SelfSupDataPreprocessor', 'RelativeLocDataPreprocessor',
    'RotationPredDataPreprocessor', 'CAEDataPreprocessor',
    'VideoMAEDataPreprocessor', 'ResLayerExtraNorm', 'NormEMAVectorQuantizer',
    'TwoNormDataPreprocessor', 'PromptTransformerEncoderLayer',
    'build_clip_model', 'build_1d_sincos_position_embedding'
]
