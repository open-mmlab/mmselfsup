# Copyright (c) OpenMMLab. All rights reserved.
from mmselfsup.registry import DATASETS


def build_dataset(cfg):
    """Build dataset."""
    return DATASETS.build(cfg)
