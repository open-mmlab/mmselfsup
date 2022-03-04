# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import ANY

import pytest

from mmselfsup.datasets import (ConcatDataset, DeepClusterDataset,
                                RepeatDataset, build_dataloader, build_dataset)

DATASET_CONFIG = dict(
    type='DeepClusterDataset',
    data_source=dict(
        type='ImageNet',
        data_prefix=ANY,
        ann_file=osp.join('tests', 'data', 'data_list.txt'),
    ),
    pipeline=[
        dict(type='RandomResizedCrop', size=224),
        dict(type='RandomHorizontalFlip'),
        dict(type='RandomRotation', degrees=2),
        dict(
            type='ColorJitter',
            brightness=0.4,
            contrast=0.4,
            saturation=1.0,
            hue=0.5),
        dict(type='RandomGrayscale', p=0.2),
    ],
)


@pytest.mark.parametrize('cfg, expected_type', [
    [
        [
            DATASET_CONFIG,
            DATASET_CONFIG,
        ],
        ConcatDataset,
    ],
    [
        {
            'type': 'RepeatDataset',
            'times': 3,
            'dataset': DATASET_CONFIG
        },
        RepeatDataset,
    ],
    [
        DATASET_CONFIG,
        DeepClusterDataset,
    ],
])
def test_build_dataset(cfg, expected_type):
    assert isinstance(build_dataset(cfg), expected_type)


def test_build_dataloader():
    dataset = build_dataset(DATASET_CONFIG)

    with pytest.raises(ValueError):
        data_loader = build_dataloader(dataset)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        samples_per_gpu=None,
        dist=False,
    )
    assert len(data_loader) == 2
    assert data_loader.batch_size == 1
