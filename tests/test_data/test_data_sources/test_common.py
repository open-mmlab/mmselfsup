# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest.mock import MagicMock

import pytest

from mmselfsup.datasets import DATASOURCES


@pytest.mark.parametrize('dataset_name',
                         ['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageList'])
def test_data_sources_override_default(dataset_name):
    dataset_class = DATASOURCES.get(dataset_name)
    load_annotations_f = dataset_class.load_annotations
    dataset_class.load_annotations = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    dataset = dataset_class(data_prefix='', classes=('bus', 'car'))
    assert dataset.CLASSES == ('bus', 'car')

    # Test setting classes as a list
    dataset = dataset_class(data_prefix='', classes=['bus', 'car'])
    assert dataset.CLASSES == ['bus', 'car']

    # Test setting classes through a file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + 'classes.txt'
        with open(path, 'w') as f:
            f.write('bus\ncar\n')
    dataset = dataset_class(data_prefix='', classes=path)
    assert dataset.CLASSES == ['bus', 'car']

    # Test overriding not a subset
    dataset = dataset_class(data_prefix='', classes=['foo'])
    assert dataset.CLASSES == ['foo']

    # Test default behavior
    dataset = dataset_class(data_prefix='')
    assert dataset.data_prefix == ''
    assert dataset.ann_file is None
    assert dataset.CLASSES == original_classes

    dataset_class.load_annotations = load_annotations_f
