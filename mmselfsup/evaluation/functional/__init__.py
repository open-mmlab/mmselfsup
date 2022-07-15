# Copyright (c) OpenMMLab. All rights reserved.
from .knn_classifier import knn_classifier

try:
    from .res_layer_extra_norm import ResLayerExtraNorm
except ImportError:
    ResLayerExtraNorm = None

__all__ = ['knn_classifier', 'ResLayerExtraNorm']
