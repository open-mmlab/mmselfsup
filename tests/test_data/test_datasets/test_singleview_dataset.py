# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest

from mmselfsup.datasets import SingleViewDataset

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


def test_single_view_dataset():
    data = dict(
        data_source=dict(
            type=data_source,
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        pipeline=train_pipeline,
        prefetch=prefetch)
    dataset = SingleViewDataset(**data)
    x = dataset[0]
    assert 'img' in x
    assert 'label' in x
    assert 'idx' in x
    assert x['img'].size() == (3, 4, 4)
    assert x['idx'] == 0

    fake_results = {'test': np.array([[0.7, 0, 0.3], [0.5, 0.3, 0.2]])}

    with pytest.raises(AssertionError):
        eval_res = dataset.evaluate({'test': np.array([[0.7, 0, 0.3]])},
                                    topk=(1))

    eval_res = dataset.evaluate(fake_results, topk=(1, 2))
    assert eval_res['test_top1'] == 1 * 100.0 / 2
    assert eval_res['test_top2'] == 2 * 100.0 / 2
