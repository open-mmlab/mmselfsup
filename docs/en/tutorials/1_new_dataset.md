# Tutorial 1: Adding New Dataset

In this tutorial, we introduce the basic steps to create your customized dataset:

- [Tutorial 1: Adding New Dataset](#tutorial-1-adding-new-dataset)
  - [An example of customized dataset](#an-example-of-customized-dataset)
  - [Creating the `DataSource`](#creating-the-datasource)
  - [Creating the `Dataset`](#creating-the-dataset)
  - [Modify config file](#modify-config-file)

If your algorithm does not need any customized dataset, you can use these off-the-shelf datasets under [datasets](../../mmselfsup/datasets). But to use these existing datasets, you have to convert your dataset to existing dataset format.

### An example of customized dataset

Assuming the format of your dataset's annotation file is:

```text
000001.jpg 0
000002.jpg 1
```

To write a new dataset, you need to implement:

- `DataSource`: inherited from `BaseDataSource` and responsible for loading the annotation files and reading images.
- `Dataset`: inherited from `BaseDataset` and responsible for applying transformation to images and packing these images.

### Creating the `DataSource`

Assume the name of your `DataSource` is `NewDataSource`, you can create a file, named `new_data_source.py` under `mmselfsup/datasets/data_sources` and implement `NewDataSource` in it.

```python
import mmcv
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class NewDataSource(BaseDataSource):

    def load_annotations(self):

        assert isinstance(self.ann_file, str)
        data_infos = []
        # writing your code here.
        return data_infos
```

Then, add `NewDataSource` in `mmselfsup/dataset/data_sources/__init__.py`.

```python
from .base import BaseDataSource
...
from .new_data_source import NewDataSource

__all__ = [
    'BaseDataSource', ..., 'NewDataSource'
]
```

### Creating the `Dataset`

Assume the name of your `Dataset` is `NewDataset`, you can create a file, named `new_dataset.py` under `mmselfsup/datasets` and implement `NewDataset` in it.

```python
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class NewDataset(BaseDataset):

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        # writing your code here
    def __getitem__(self, idx):
        # writing your code here
        return dict(img=img)

    def evaluate(self, results, logger=None):
        return NotImplemented
```

Then, add `NewDataset` in `mmselfsup/dataset/__init__.py`.

```python
from .base import BaseDataset
...
from .new_dataset import NewDataset

__all__ = [
    'BaseDataset', ..., 'NewDataset'
]
```

### Modify config file

To use `NewDataset`, you can modify the config as the following:

```python
train=dict(
        type='NewDataset',
        data_source=dict(
            type='NewDataSource',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))

```
