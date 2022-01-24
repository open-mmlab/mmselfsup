import pytest

from mmselfsup.datasets import (ConcatDataset, MultiViewDataset, RepeatDataset,
                                build_dataset)

DATASET_CONFIG = {
    'type': 'MultiViewDataset',
    'data_source': {
        'type': 'ImageNet',
        'data_prefix': 'data/imagenet/train',
        'ann_file': 'data/imagenet/meta/train.txt',
    },
}


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
        MultiViewDataset,
    ],
])
def test_build_dataset(cfg, expected_type):
    assert isinstance(build_dataset(cfg), expected_type)
