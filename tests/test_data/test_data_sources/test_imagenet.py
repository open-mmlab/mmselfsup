# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmselfsup.datasets.data_sources import ImageNet


def test_imagenet():
    data_source = dict(data_prefix='tests')

    dataset = ImageNet(**data_source)
    assert len(dataset) == 2

    with pytest.raises(TypeError):
        dataset = ImageNet(ann_file=1, **data_source)

    with pytest.raises(RuntimeError):
        dataset = ImageNet(data_prefix=osp.join(osp.dirname(__file__)))
