# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmselfsup.datasets.data_sources import ImageList


def test_image_list():
    data_source = dict(
        data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
        ann_file=osp.join(
            osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
    )

    dataset = ImageList(**data_source)
    assert len(dataset) == 2

    with pytest.raises(AssertionError):
        dataset = ImageList(
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'))
