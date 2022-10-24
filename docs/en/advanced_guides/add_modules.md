# Add Modules

In this tutorial, we introduce the basic steps to create your customized modules. Before learning to create your customized modules, it is recommended to learn the basic concept of models in file [models.md](models.md). You can customize all the components introduced in [models.md](models.md), such as **backbone**, **neck**, **head** and **loss**.

- [Add Modules](#add-modules)
  - [Add a new backbone](#add-a-new-backbone)
  - [Add a new neck](#add-a-new-neck)
  - [Add a new head](#add-a-new-head)
  - [Add a new loss](#add-a-new-loss)
  - [Combine all](#combine-all)

## Add a new backbone

Assume you are going to create a new backbone `NewBackbone`.

1. Create a new file `mmselfsup/models/backbones/new_backbone.py` and implement `NewBackbone` in it.

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

2. Import the new backbone module in `mmselfsup/models/backbones/__init__.py`.

```python
...
from .new_backbone import NewBackbone

__all__ = [
    ...,
    'NewBackbone',
    ...
]
```

3. Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='NewBackbone',
        ...),
    ...
)
```

## Add a new neck

You can write a new neck inherited from `BaseModule` from mmengine, and overwrite `forward`. We have a unified interface for weight initialization in mmengine, you can use `init_cfg` to specify the initialization function and arguments, or overwrite `init_weights` if you prefer customized initialization.

We include all necks in `mmselfsup/models/necks`. Assume you are going to create a new neck `NewNeck`.

1. Create a new file `mmselfsup/models/necks/new_neck.py` and implement `NewNeck` in it.

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

You need to implement the `forward` function, which applies some operations on the output from the backbone and forwards the results to the head.

2. Import the new neck module in `mmselfsup/models/necks/__init__.py`.

```python
...
from .new_neck import NewNeck

__all__ = [
    ...,
    'NewNeck',
    ...
]
```

3. Use it in your config file.

```python
model = dict(
    ...
    neck=dict(
        type='NewNeck',
        ...),
    ...
)
```

## Add a new head

You can write a new head inherited from `BaseModule` from mmengine, and overwrite `forward`.

We include all heads in `mmselfsup/models/heads`. Assume you are going to create a new head `NewHead`.

1. Create a new file `mmselfsup/models/heads/new_head.py` and implement `NewHead` in it.

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

You need to implement the `forward` function, which applies some operations on the output from the neck/backbone and computes the loss. Please note that the loss module should be built in the head module for the loss computation.

2. Import the new head module in `mmselfsup/models/heads/__init__.py`.

```python
...
from .new_head import NewHead

__all__ = [
    ...,
    'NewHead',
    ...
]
```

3. Use it in your config file.

```python
model = dict(
    ...
    head=dict(
        type='NewHead',
        ...),
    ...
)
```

## Add a new loss

To add a new loss function, we mainly implement the `forward` function in the loss module. We should register the loss module as `MODELS` as well.

We include all losses in `mmselfsup/models/losses`. Assume you are going to create a new loss `NewLoss`.

1. Create a new file `mmselfsup/models/losses/new_loss.py` and implement `NewLoss` in it.

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

2. Import the new loss module in `mmselfsup/models/losses/__init__.py`

```python
...
from .new_loss import NewLoss

__all__ = [
    ...,
    'NewLoss',
    ...
]
```

3. Use it in your config file.

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

## Combine all

After creating each component mentioned above, we need to create a new algorithm `NewAlgorithm` to organize them logically. `NewAlgorithm` takes raw images as inputs and outputs the loss to the optimizer.

1. Create a new file `mmselfsup/models/algorithms/new_algorithm.py` and implement `NewAlgorithm` in it.

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

2. Import the new algorithm module in `mmselfsup/models/algorithms/__init__.py`

```python
...
from .new_algorithm import NewAlgorithm

__all__ = [
    ...,
    'NewAlgorithm',
    ...
]
```

3. Use it in your config file.

```python
model = dict(
    type='NewAlgorithm',
    backbone=...,
    neck=...,
    head=...,
    ...
)
```
