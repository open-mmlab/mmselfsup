from .backbones import *  # noqa: F401,F403
from .builder import (build_backbone, build_model, build_head, build_loss)
from .heads import *
from .classification import Classification
from .deepcluster import DeepCluster
from .odc import ODC
from .losses import *  # noqa: F401,F403
from .necks import *
from .npid import NPID
from .memories import *
from .moco import MOCO
from .registry import (BACKBONES, MODELS, NECKS, MEMORIES, HEADS, LOSSES)
from .rotation_pred import RotationPred
from .simclr import SimCLR

#__all__ = [
#    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
#    'DETECTORS', 'CLASSIFIERS', 'build_backbone', 'build_neck', 'build_roi_extractor',
#    'build_shared_head', 'build_head', 'build_loss', 'build_detector', 'build_detector'
#]
