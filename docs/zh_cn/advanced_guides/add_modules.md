# 添加模块

在本教程中，我们将要介绍创建用户自定义模块的基本步骤。在学习创建自定义模块之前，建议先了解一下文件 [models.md](models.md)中模型的基本的概念。您可以自定义 [models.md](models.md)文件中涉及的所有模型组件，例如**主干网络 (backbone)**, **neck**, **head** 和**损失 (loss)**。

- [添加模块](#添加模块)
  - [添加新的主干网络](#添加新的主干网络)
  - [添加新的 neck](#添加新的-neck)
  - [添加新的 head](#添加新的-head)
  - [添加新的损失](#添加新的损失)
  - [组合起来](#组合起来)

## 添加新的主干网络

假如要创建新的主干网络 `NewBackbone`。

1. 新建一个文件 `mmselfsup/models/backbones/new_backbone.py` 并在其中实现新的主干网络 `NewBackbone`。

```python
import torch.nn as nn

from mmselfsup.registry import MODELS


@MODELS.register_module()
class NewBackbone(nn.Module):

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self):
        pass

    def train(self, mode=True):
        pass
```

2. 导入新添加的主干网络到 `mmselfsup/models/backbones/__init__.py`。

```python
...
from .new_backbone import NewBackbone

__all__ = [
    ...,
    'NewBackbone',
    ...
]
```

3. 在配置文件中使用自定义的主干网络。

```python
model = dict(
    ...
    backbone=dict(
        type='NewBackbone',
        ...),
    ...
)
```

## 添加新的 neck

您可以通过继承 mmengine 中的 `BaseModule` 来创建新的 neck，并且重定义其中的 `forward` 函数。我们在 mmengine 中有统一的权重初始化接口，您可以使用 `init_cfg` 来指定初始化函数和参数，或者可以重定义函数 `init_weights` 如果您希望使用自定义初始化方式。

所有的已有的 neck 都在 `mmselfsup/models/necks` 中。假如您要创建新的 neck `NewNeck`。

1. 新建一个文件 `mmselfsup/models/necks/new_neck.py` 并在其中实现 `NewNeck`。

```python
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class NewNeck(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        pass
```

您需要在 `forward` 函数中实现一些针对主干网络输出的操作，并将结果给 head。

2. 导入新定义的 neck 模块到 `mmselfsup/models/necks/__init__.py`。

```python
...
from .new_neck import NewNeck

__all__ = [
    ...,
    'NewNeck',
    ...
]
```

3. 在配置文件中使用自定义的 neck 模块。

```python
model = dict(
    ...
    neck=dict(
        type='NewNeck',
        ...),
    ...
)
```

## 添加新的 head

您可以通过继承 mmengine 中的 `BaseModule` 来创建新的 head，并且重定义其中的 `forward` 函数。

所有已有的 head 都在 `mmselfsup/models/heads` 文件中。假如您想创建新的head `NewHead`。

1. 创建文件 `mmselfsup/models/heads/new_head.py` 并在其中实现 `NewHead`。

```python
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class NewHead(BaseModule):

    def __init__(self, loss, **kwargs):
        super().__init__()
        # build loss
        self.loss = MODELS.build(loss)
        # other specific initializations

    def forward(self, *args, **kwargs):
        pass
```

您需要在 `forward` 函数中实现一些针对主干网络或 neck 输出的操作，并计算损失。请注意，损失模块应构建在 head 模块中以进行损失计算。

2. 在 `mmselfsup/models/heads/__init__.py` 中导入新创建的 head 模块。

```python
...
from .new_head import NewHead

__all__ = [
    ...,
    'NewHead',
    ...
]
```

3. 在配置文件中使用自定义的 head。

```python
model = dict(
    ...
    head=dict(
        type='NewHead',
        ...),
    ...
)
```

## 添加新的损失

添加新的损失函数时，主要需要在 loss 模块中实现 `forward` 函数。同时您需要将 loss 模块也注册 (register) 为 `MODELS`。

所有已有的损失函数都在 `mmselfsup/models/losses` 中。假如您想创建新的损失 `NewLoss`。

1. 创建文件 `mmselfsup/models/losses/new_loss.py` 并在其中实现 `NewLoss`。

```python
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class NewLoss(BaseModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, *args, **kwargs):
        pass
```

2. 在 `mmselfsup/models/losses/__init__.py` 中导入新定义的 loss 模块。

```python
...
from .new_loss import NewLoss

__all__ = [
    ...,
    'NewLoss',
    ...
]
```

3. 在配置文件中使用自定义的 loss 模块。

```python
model = dict(
    ...
    head=dict(
        ...
        loss=dict(
            type='NewLoss',
            ...),
        ...),
    ...
)
```

## 组合起来

在创建好上述的各个模块组件之后，我们需要创建一个新的算法 `NewAlgorithm` 来将各个组件按照逻辑顺序组合起来。`NewAlgorithm` 使用原始图像作为输入并输出损失函数值给优化器 (optimizer)。

1. 创建文件 `mmselfsup/models/algorithms/new_algorithm.py` 并在其中实现 `NewAlgorithm`。

```python
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class NewAlgorithm(BaseModel):

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super().__init__(init_cfg)
        pass

    def extract_feat(self, inputs, **kwargs):
        pass

    def loss(self, inputs, data_samples, **kwargs):
        pass

    def predict(self, inputs, data_samples, **kwargs):
        pass
```

2. 在 `mmselfsup/models/algorithms/__init__.py` 中导入新创建的算法模块 `NewAlgorithm`。

```python
...
from .new_algorithm import NewAlgorithm

__all__ = [
    ...
    'NewAlgorithm',
    ...
]
```

3. 在配置文件中使用自定义的算法模块。

```python
model = dict(
    type='NewAlgorithm',
    backbone=...,
    neck=...,
    head=...,
    ...
)
```
