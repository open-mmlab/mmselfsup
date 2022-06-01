# 教程 3：添加新的模块

- [教程 3：添加新的模块](#%E6%95%99%E7%A8%8B-3-%E6%B7%BB%E5%8A%A0%E6%96%B0%E7%9A%84%E6%A8%A1%E5%9D%97)
  - [添加新的 backbone](#%E6%B7%BB%E5%8A%A0%E6%96%B0%E7%9A%84-backbone)
  - [添加新的 Necks](#%E6%B7%BB%E5%8A%A0%E6%96%B0%E7%9A%84-Necks)
  - [添加新的损失](#%E6%B7%BB%E5%8A%A0%E6%96%B0%E7%9A%84%E6%8D%9F%E5%A4%B1)
  - [合并所有改动](#%E5%90%88%E5%B9%B6%E6%89%80%E6%9C%89%E6%94%B9%E5%8A%A8)

在自监督学习领域，每个模型可以被分为以下四个部分：

- backbone：用于提取图像特征。
- projection head：将 backbone 提取的特征映射到另一空间。
- loss：用于模型优化的损失函数。
- memory bank（可选）：一些方法（例如 `odc` ），需要额外的 memory bank 用于存储图像特征。

## 添加新的 backbone

假设我们要创建一个自定义的 backbone `CustomizedBackbone`。

1.创建新文件 `mmselfsup/models/backbones/customized_backbone.py` 并在其中实现 `CustomizedBackbone` 。

```python
import torch.nn as nn
from ..builder import BACKBONES

@BACKBONES.register_module()
class CustomizedBackbone(nn.Module):

    def __init__(self, **kwargs):

        ## TODO

    def forward(self, x):

        ## TODO

    def init_weights(self, pretrained=None):

        ## TODO

    def train(self, mode=True):

        ## TODO
```

2.在 `mmselfsup/models/backbones/__init__.py` 中导入自定义的 backbone。

```python
from .customized_backbone import CustomizedBackbone

__all__ = [
    ..., 'CustomizedBackbone'
]
```

3.在你的配置文件中使用它。

```python
model = dict(
    ...
    backbone=dict(
        type='CustomizedBackbone',
        ...),
    ...
)
```

## 添加新的 Necks

我们在 `mmselfsup/models/necks` 中包含了所有的 projection heads。假设我们要创建一个 `CustomizedProjHead` 。

1.创建一个新文件 `mmselfsup/models/necks/customized_proj_head.py` 并在其中实现 `CustomizedProjHead` 。

```python
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class CustomizedProjHead(BaseModule):

    def __init__(self, *args, **kwargs):
        super(CustomizedProjHead, self).__init__(init_cfg)
        ## TODO
    def forward(self, x):
        ## TODO
```

你需要实现前向函数，该函数从 backbone 中获取特征，并输出映射后的特征。

2.在 `mmselfsup/models/necks/__init__` 中导入 `CustomizedProjHead` 。

```python
from .customized_proj_head import CustomizedProjHead

__all__ = [
    ...,
    CustomizedProjHead,
    ...
]
```

3.在你的配置文件中使用它。

```python
model = dict(
    ...,
    neck=dict(
        type='CustomizedProjHead',
        ...),
   ...)
```

## 添加新的损失

为了增加一个新的损失函数，我们主要在损失模块中实现 `forward` 函数。

1.创建一个新的文件 `mmselfsup/models/heads/customized_head.py` 并在其中实现你自定义的 `CustomizedHead` 。

```python
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class CustomizedHead(BaseModule):

    def __init__(self, *args, **kwargs):
        super(CustomizedHead, self).__init__()

        ## TODO

    def forward(self, *args, **kwargs):

        ## TODO
```

2.在 `mmselfsup/models/heads/__init__.py` 中导入该模块。

```python
from .customized_head import CustomizedHead

__all__ = [..., CustomizedHead, ...]
```

3.在你的配置文件中使用它。

```python
model = dict(
    ...,
    head=dict(type='CustomizedHead')
    )
```

## 合并所有改动

在创建了上述每个组件后，我们需要创建一个 `CustomizedAlgorithm` 来有逻辑的将他们组织到一起。 `CustomizedAlgorithm` 接收原始图像作为输入，并将损失输出给优化器。

1.创建一个新文件 `mmselfsup/models/algorithms/customized_algorithm.py` 并在其中实现 `CustomizedAlgorithm`。

```python
# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import GatherLayer
from .base import BaseModel


@ALGORITHMS.register_module()
class CustomizedAlgorithm(BaseModel):

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(SimCLR, self).__init__(init_cfg)

        ## TODO

    def forward_train(self, img, **kwargs):

        ## TODO
```

2.在 `mmselfsup/models/algorithms/__init__.py` 中导入该模块。

```python
from .customized_algorithm import CustomizedAlgorithm

__all__ = [..., CustomizedAlgorithm, ...]
```

3.在你的配置文件中使用它。

```python
model = dict(
    type='CustomizedAlgorightm',
    backbone=...,
    neck=...,
    head=...)
```
