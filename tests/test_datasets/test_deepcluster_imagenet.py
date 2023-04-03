# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
from mmengine.registry import init_default_scope

from mmselfsup.datasets import DeepClusterImageNet

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=4)
]


def test_deepcluster_dataset():
    init_default_scope('mmselfsup')

    data = dict(
        ann_file=osp.join(
            osp.dirname(__file__), '..', 'data', 'data_list.txt'),
        metainfo=None,
        data_root=osp.join(osp.dirname(__file__), '..', 'data'),
        pipeline=train_pipeline)
    dataset = DeepClusterImageNet(**data)
    assert len(dataset.clustering_labels) == 2

    x = dataset[0]
    print(x)
    assert x['img'].shape == (4, 4, 3)
    assert x['clustering_label'] == -1
    assert x['sample_idx'] == 0

    with pytest.raises(AssertionError):
        dataset.assign_labels([1])

    dataset.assign_labels([1, 0])
    assert dataset.clustering_labels[0] == 1
    assert dataset.clustering_labels[1] == 0

    x = dataset[0]
    assert x['clustering_label'] == 1
