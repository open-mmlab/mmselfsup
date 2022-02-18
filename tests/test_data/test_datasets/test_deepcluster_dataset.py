# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmselfsup.datasets import DeepClusterDataset

# dataset settings
data_source = 'ImageNet'
dataset_type = 'DeepClusterDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [dict(type='RandomResizedCrop', size=4)]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])


def test_deepcluster_dataset():
    data = dict(
        data_source=dict(
            type=data_source,
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        pipeline=train_pipeline,
        prefetch=prefetch)
    dataset = DeepClusterDataset(**data)
    x = dataset[0]
    assert x['img'].size() == (3, 4, 4)
    assert x['pseudo_label'] == -1
    assert x['idx'] == 0

    with pytest.raises(AssertionError):
        dataset.assign_labels([1])

    dataset.assign_labels([1, 0])
    assert dataset.clustering_labels[0] == 1
    assert dataset.clustering_labels[1] == 0

    x = dataset[0]
    assert x['pseudo_label'] == 1
