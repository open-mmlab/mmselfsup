# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmselfsup.datasets.transforms import PackSelfSupInputs


def test_pack_selfsup_inputs():
    transform = PackSelfSupInputs(
        key='img',
        algorithm_keys=['gt_label', 'pred_label', 'sample_idx', 'mask'])

    # image with 3 channels
    results = {
        'img': np.ones((8, 8, 3)),
        'gt_label': 1,
        'pred_label': 1,
        'sample_idx': 1,
        'mask': np.ones((2, 2))
    }
    results = transform(results)
    assert list(results['inputs'][0].shape) == [3, 8, 8]
    assert results['data_samples'].gt_label.value == torch.tensor([1])
    assert results['data_samples'].pred_label.value == torch.tensor([1])
    assert results['data_samples'].sample_idx.value == torch.tensor([1])
    assert list(results['data_samples'].mask.value.shape) == [2, 2]

    # image with 1 channel
    transform = PackSelfSupInputs(key='img', algorithm_keys=['gt_label'])
    results = {'img': np.ones((8, 8)), 'gt_label': 1}
    results = transform(results)
    assert list(results['inputs'][0].shape) == [1, 8, 8]
    assert results['data_samples'].gt_label.value == torch.tensor([1])

    # img is a list
    transform = PackSelfSupInputs(key='img', algorithm_keys=['gt_label'])
    results = {'img': [np.ones((8, 8))], 'gt_label': 1}
    results = transform(results)
    assert list(results['inputs'][0].shape) == [1, 8, 8]
    assert results['data_samples'].gt_label.value == torch.tensor([1])

    # pseudo_label_keys is not None
    transform = PackSelfSupInputs(key='img', pseudo_label_keys=['angle'])
    results = {'img': [np.ones((8, 8))], 'angle': 90}
    results = transform(results)
    assert results['data_samples'].pseudo_label.angle == torch.tensor([90])

    # test repr
    assert isinstance(str(transform), str)
