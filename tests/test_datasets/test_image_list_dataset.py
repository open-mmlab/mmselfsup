# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from mmengine.registry import init_default_scope

from mmselfsup.datasets import ImageList

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=4)
]


def test_image_list_dataset():
    init_default_scope('mmselfsup')

    data = dict(
        ann_file='',
        metainfo=None,
        data_root=osp.join(osp.dirname(__file__), '..', 'data'),
        pipeline=train_pipeline)
    with pytest.raises(AssertionError):
        dataset = ImageList(**data)

    ann_file = osp.join(
        osp.dirname(__file__), '..', 'data', 'data_list_no_label.txt')
    data = dict(
        ann_file=ann_file,
        metainfo=None,
        data_root=osp.join(osp.dirname(__file__), '..', 'data'),
        pipeline=train_pipeline)
    dataset = ImageList(**data)
    assert len(dataset) == 2
    assert dataset[0]['gt_label'] == np.array(-1)
