# Copyright (c) OpenMMLab. All rights reserved.
from mmselfsup.registry import MODELS

ALGORITHMS = MODELS
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
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


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_memory(cfg):
    """Build memory."""
    return MEMORIES.build(cfg)
