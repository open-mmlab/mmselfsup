import pytest

from mmselfsup.datasets import (ConcatDataset, MultiViewDataset, RepeatDataset,
                                build_dataset)

DATASET_CONFIG = dict(
    type='MultiViewDataset',
    data_source=dict(
        type='ImageNet',
        data_prefix='data/imagenet/train',
        ann_file='data/imagenet/meta/train.txt',
    ),
    num_views=[1, 1],
    pipelines=[
        dict(type='RandomResizedCrop', size=224, interpolation=3),
        dict(type='RandomHorizontalFlip'),
        dict(
            type='RandomAppliedTrans',
            transforms=[
                dict(
                    type='ColorJitter',
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1)
            ],
            p=0.8),
        dict(type='RandomGrayscale', p=0.2),
        dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=1.),
        dict(type='Solarization', p=0.),
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
        MultiViewDataset,
    ],
])
def test_build_dataset(cfg, expected_type):
    assert isinstance(build_dataset(cfg), expected_type)
