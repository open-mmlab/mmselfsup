# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

ALGORITHMS = MODELS
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
MEMORIES = MODELS


def build_algorithm(cfg):
    """Build algorithm."""
    return ALGORITHMS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_memory(cfg):
    """Build memory."""
    return MEMORIES.build(cfg)
