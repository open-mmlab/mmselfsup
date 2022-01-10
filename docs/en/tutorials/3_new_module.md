# Tutorial 3: Adding New Modules

- [Tutorial 3: Adding New Modules](#tutorial-3-adding-new-modules)
  - [Add new backbone](#add-new-backbone)
  - [Add new necks](#add-new-necks)
  - [Add new loss](#add-new-loss)
  - [Combine all](#combine-all)

In self-supervised learning domain, each model can be divided into following four parts:

- backbone: used to extract image's feature
- projection head: projects feature extracted by backbone to another space
- loss: loss function the model will optimize
- memory bank(optional): some methods, `e.g. odc`, need extract memory bank to store image's feature.

## Add new backbone

Assuming we are going to create a customized backbone `CustomizedBackbone`

1.Create a new file `mmselfsup/models/backbones/customized_backbone.py` and implement `CustomizedBackbone` in it.

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

2.Import the customized backbone in `mmselfsup/models/backbones/__init__.py`.

```python
from .customized_backbone import CustomizedBackbone

__all__ = [
    ..., 'CustomizedBackbone'
]
```

3.Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='CustomizedBackbone',
        ...),
    ...
)
```

## Add new necks

we include all projection heads in `mmselfsup/models/necks`. Assuming we are going to create a `CustomizedProjHead`.

1.Create a new file `mmselfsup/models/necks/customized_proj_head.py` and implement `CustomizedProjHead` in it.

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

You need to implement the forward function, which takes the feature from the backbone and outputs the projected feature.

2.Import the `CustomizedProjHead` in `mmselfsup/models/necks/__init__`.

```python
from .customized_proj_head import CustomizedProjHead

__all__ = [
    ...,
    CustomizedProjHead,
    ...
]
```

3.Use it in your config file.

```python
model = dict(
    ...,
    neck=dict(
        type='CustomizedProjHead',
        ...),
   ...)
```

## Add new loss

To add a new loss function, we mainly implement the `forward` function in the loss module.

1.Create a new file `mmselfsup/models/heads/customized_head.py` and implement your customized `CustomizedHead` in it.

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

2.Import the module in `mmselfsup/models/heads/__init__.py`

```python
from .customized_head import CustomizedHead

__all__ = [..., CustomizedHead, ...]
```

3.Use it in your config file.

```python
model = dict(
    ...,
    head=dict(type='CustomizedHead')
    )
```

## Combine all

After creating each component, mentioned above, we need to create a `CustomizedAlgorithm` to organize them logically. And the `CustomizedAlgorithm` takes raw images as inputs and outputs the loss to the optimizer.

1.Create a new file `mmselfsup/models/algorithms/customized_algorithm.py` and implement `CustomizedAlgorithm` in it.

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

2.Import the module in `mmselfsup/models/algorithms/__init__.py`

```python
from .customized_algorithm import CustomizedAlgorithm

__all__ = [..., CustomizedAlgorithm, ...]
```

3.Use it in your config file.

```python
model = dict(
    type='CustomizedAlgorightm',
    backbone=...,
    neck=...,
    head=...)
```
