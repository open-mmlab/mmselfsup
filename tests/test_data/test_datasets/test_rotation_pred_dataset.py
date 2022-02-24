# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np

from mmselfsup.datasets import RotationPredDataset

# dataset settings
data_source = 'ImageNet'
dataset_type = 'RotationPredDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [dict(type='RandomResizedCrop', size=4)]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])


def test_rotation_pred_dataset():
    # prefetch False
    data = dict(
        data_source=dict(
            type=data_source,
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        pipeline=train_pipeline,
        prefetch=prefetch)
    dataset = RotationPredDataset(**data)
    x = dataset[0]
    assert x['img'].size() == (4, 3, 4, 4)
    assert (x['rot_label'].numpy() == np.array([0, 1, 2, 3])).all()
