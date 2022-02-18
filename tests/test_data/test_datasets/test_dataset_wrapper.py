# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import MagicMock, patch

from mmselfsup.datasets import BaseDataset, ConcatDataset, RepeatDataset


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_dataset():
    BaseDataset.CLASSES = ('foo', 'bar')
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    data = dict(
        data_source=dict(
            type='ImageNet',
            data_prefix=osp.join(osp.dirname(__file__), '..', '..', 'data'),
            ann_file=osp.join(
                osp.dirname(__file__), '..', '..', 'data', 'data_list.txt'),
        ),
        pipeline=[])
    dataset = BaseDataset(**data)
    dataset.data_infos = MagicMock()
    return dataset


def test_concat_dataset():
    dataset_a = construct_toy_dataset()
    dataset_b = construct_toy_dataset()

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[0] == 0
    assert concat_dataset[3] == 1
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)


def test_repeat_dataset():
    dataset = construct_toy_dataset()

    repeat_dataset = RepeatDataset(dataset, 10)
    assert repeat_dataset[5] == 1
    assert repeat_dataset[10] == 0
    assert len(repeat_dataset) == 10 * len(dataset)
