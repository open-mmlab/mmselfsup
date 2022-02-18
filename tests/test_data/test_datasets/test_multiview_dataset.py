# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmselfsup.datasets import MultiViewDataset

# dataset settings
data_source = 'ImageNet'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [dict(type='RandomResizedCrop', size=4)]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])


def test_multi_views_dataste():
    data = dict(
        data_source=dict(
            type=data_source,
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        num_views=[2],
        pipelines=[train_pipeline, train_pipeline],
        prefetch=prefetch)
    with pytest.raises(AssertionError):
        dataset = MultiViewDataset(**data)

    # test dataset
    data = dict(
        data_source=dict(
            type=data_source,
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        num_views=[2, 6],
        pipelines=[train_pipeline, train_pipeline],
        prefetch=prefetch)
    dataset = MultiViewDataset(**data)
    x = dataset[0]
    assert isinstance(x['img'], list)
    assert len(x['img']) == 8
