# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.data import LabelData

from mmselfsup.datasets.pipelines import PackSelfSupInputs


def test_pack_selfsup_inputs():
    transform = PackSelfSupInputs(key='img', meta_keys=['gt_label'])

    # image with 3 channels
    results = {
        'img': np.ones((8, 8, 3)),
        'gt_label': LabelData(**{'value': [1]})
    }
    results = transform(results)
    assert list(results['inputs'][0].shape) == [3, 8, 8]
    assert results['data_sample'].gt_label.value == [1]

    # image with 1 channel
    results = {'img': np.ones((8, 8)), 'gt_label': LabelData(**{'value': [1]})}
    results = transform(results)
    assert list(results['inputs'][0].shape) == [1, 8, 8]
    assert results['data_sample'].gt_label.value == [1]

    # img is a list
    results = {
        'img': [np.ones((8, 8))],
        'gt_label': LabelData(**{'value': [1]})
    }
    results = transform(results)
    assert list(results['inputs'][0].shape) == [1, 8, 8]
    assert results['data_sample'].gt_label.value == [1]

    # test repr
    assert isinstance(str(transform), str)
