# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np

from mmselfsup.datasets import RelativeLocDataset

# dataset settings
data_source = 'ImageNet'
dataset_type = 'RelativeLocDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [dict(type='RandomResizedCrop', size=224)]
# prefetch
format_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]


def test_relative_loc_dataset():
    # prefetch False
    data = dict(
        data_source=dict(
            type=data_source,
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        pipeline=train_pipeline,
        format_pipeline=format_pipeline)
    dataset = RelativeLocDataset(**data)
    x = dataset[0]
    split_per_side = 3
    patch_jitter = 21
    h_grid = 224 // split_per_side
    w_grid = 224 // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    assert x['img'].size() == (8, 6, h_patch, w_patch)
    assert (x['patch_label'].numpy() == np.array([0, 1, 2, 3, 4, 5, 6,
                                                  7])).all()
