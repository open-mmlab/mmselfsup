from torch import nn

from openselfsup.utils import build_from_cfg
from .registry import (BACKBONES, MODELS, NECKS, HEADS, MEMORIES, LOSSES)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_memory(cfg):
    return build(cfg, MEMORIES)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_model(cfg):
    return build(cfg, MODELS)
