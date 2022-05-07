# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
# TODO: will use real PixelData once it is added in mmengine
from mmengine.data import InstanceData

from mmselfsup.core import SelfSupDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestSelfSupDataSample(TestCase):

    def test_init(self):
        meta_info = dict(img_size=[256, 256])

        det_data_sample = SelfSupDataSample(metainfo=meta_info)
        assert 'img_size' in det_data_sample
        assert det_data_sample.img_size == [256, 256]

    def test_setter(self):
        selfsup_data_sample = SelfSupDataSample()
        # test gt_label
        gt_label_data = dict(value=[1])
        gt_label = InstanceData(**gt_label_data)
        selfsup_data_sample.gt_label = gt_label
        assert 'gt_label' in selfsup_data_sample
        assert _equal(selfsup_data_sample.gt_label.value,
                      gt_label_data['value'])

        # test idx
        idx_data = dict(value=[1])
        idx_instances = InstanceData(**idx_data)
        selfsup_data_sample.idx = idx_instances
        assert 'idx' in selfsup_data_sample
        assert _equal(selfsup_data_sample.idx.value, idx_data['value'])

        # test mask
        mask_data = dict(value=torch.rand(4, 4))
        mask = InstanceData(**mask_data)
        selfsup_data_sample.mask = mask
        assert 'mask' in selfsup_data_sample
        assert _equal(selfsup_data_sample.mask.value, mask_data['value'])

        # test pred_label
        pred_label_data = dict(value=[1])
        pred_label_instances = InstanceData(**pred_label_data)
        selfsup_data_sample.pred_label = pred_label_instances
        assert 'pred_label' in selfsup_data_sample
        assert _equal(selfsup_data_sample.pred_label.value,
                      pred_label_data['value'])

    def test_deleter(self):

        gt_label_data = dict(value=[1])
        selfsup_data_sample = SelfSupDataSample()
        gt_label = InstanceData(value=gt_label_data)
        selfsup_data_sample.gt_label = gt_label
        assert 'gt_label' in selfsup_data_sample
        del selfsup_data_sample.gt_label
        assert 'gt_label' not in selfsup_data_sample

        idx_data = dict(value=[1])
        selfsup_data_sample = SelfSupDataSample()
        idx = InstanceData(value=idx_data)
        selfsup_data_sample.idx = idx
        assert 'idx' in selfsup_data_sample
        del selfsup_data_sample.idx
        assert 'idx' not in selfsup_data_sample

        mask_data = dict(value=torch.rand(4, 4))
        selfsup_data_sample = SelfSupDataSample()
        mask = InstanceData(value=mask_data)
        selfsup_data_sample.mask = mask
        assert 'mask' in selfsup_data_sample
        del selfsup_data_sample.mask
        assert 'mask' not in selfsup_data_sample

        pred_label_data = dict(value=[1])
        selfsup_data_sample = SelfSupDataSample()
        pred_label = InstanceData(value=pred_label_data)
        selfsup_data_sample.pred_label = pred_label
        assert 'pred_label' in selfsup_data_sample
        del selfsup_data_sample.pred_label
        assert 'pred_label' not in selfsup_data_sample
