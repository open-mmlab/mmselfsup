# 教程 1: 添加新的数据格式

在本节教程中，我们将介绍创建自定义数据格式的基本步骤：

- [教程 1: 添加新的数据格式](#教程 1: 添加新的数据格式)
    - [自定义数据格式示例](#自定义数据格式示例)
    - [创建`DataSource`子类](#创建DataSource子类)
    - [创建`DataSource`子类](#创建DataSource子类)
    - [修改配置文件](#修改配置文件)

如果你的算法不需要任何定制的数据格式，你可以使用[datasets](../../mmselfsup/datasets)目录中这些现成的数据格式。但是要使用这些现有的数据格式，你必须将你的数据集转换为现有的数据格式。

### 自定义数据格式示例

假设你的数据集的注释文件格式是：

```text
000001.jpg 0
000002.jpg 1
```

要编写一个新的数据格式，你需要实现：

- 子类`DataSource`：继承自父类`BaseDataSource`——负责加载注释文件和读取图像。
- 子类`Dataset`：继承自父类 `BaseDataset` ——负责对图像进行转换和打包。

### 创建 `DataSource`子类

假设你基于父类`DataSource` 创建的子类名为 `NewDataSource`， 你可以在`mmselfsup/datasets/data_sources` 目录下创建一个文件，文件名为 `new_data_source.py` ，并在这个文件中实现 `NewDataSource` 创建。

```py
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

然后， 在 `mmselfsup/dataset/data_sources/__init__.py`中添加`NewDataSource`。

```py
from .base import BaseDataSource
...
from .new_data_source import NewDataSource

__all__ = [
    'BaseDataSource', ..., 'NewDataSource'
]
```

### 创建 `Dataset`子类

假设你基于父类 `Dataset` 创建的子类名为 `NewDataset`，你可以在`mmselfsup/datasets`目录下创建一个文件，文件名为`new_dataset.py` ，并在这个文件中实现 `NewDataset` 创建。

```py
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

然后，在 `mmselfsup/dataset/__init__.py`中添加 `NewDataset`。

```py
from .base import BaseDataset
...
from .new_dataset import NewDataset

__all__ = [
    'BaseDataset', ..., 'NewDataset'
]
```

### 修改配置文件

为了使用 `NewDataset`，你可以修改配置如下：

```py
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
