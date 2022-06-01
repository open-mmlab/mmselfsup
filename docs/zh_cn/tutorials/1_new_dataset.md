# 教程 1: 添加新的数据格式

在本节教程中，我们将介绍创建自定义数据格式的基本步骤：

- [教程 1: 添加新的数据格式](#%E6%95%99%E7%A8%8B-1-%E6%B7%BB%E5%8A%A0%E6%96%B0%E7%9A%84%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F)
  - [自定义数据格式示例](#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F%E7%A4%BA%E4%BE%8B)
  - [创建`DataSource`子类](#%E5%88%9B%E5%BB%BA-datasource-%E5%AD%90%E7%B1%BB)
  - [创建`Dataset`子类](#%E5%88%9B%E5%BB%BA-dataset-%E5%AD%90%E7%B1%BB)
  - [修改配置文件](#%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)

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

### 创建 `DataSource` 子类

假设你基于父类`DataSource` 创建的子类名为 `NewDataSource`， 你可以在`mmselfsup/datasets/data_sources` 目录下创建一个文件，文件名为 `new_data_source.py` ，并在这个文件中实现 `NewDataSource` 创建。

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

然后， 在 `mmselfsup/dataset/data_sources/__init__.py`中添加`NewDataSource`。

```python
from .base import BaseDataSource
...
from .new_data_source import NewDataSource

__all__ = [
    'BaseDataSource', ..., 'NewDataSource'
]
```

### 创建 `Dataset` 子类

假设你基于父类 `Dataset` 创建的子类名为 `NewDataset`，你可以在`mmselfsup/datasets`目录下创建一个文件，文件名为`new_dataset.py` ，并在这个文件中实现 `NewDataset` 创建。

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

然后，在 `mmselfsup/dataset/__init__.py`中添加 `NewDataset`。

```python
from .base import BaseDataset
...
from .new_dataset import NewDataset

__all__ = [
    'BaseDataset', ..., 'NewDataset'
]
```

### 修改配置文件

为了使用 `NewDataset`，你可以修改配置如下：

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
