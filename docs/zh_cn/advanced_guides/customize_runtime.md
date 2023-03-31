# 自定义运行

- [自定义运行](#自定义运行)
  - [循环(Loop)](<#循环(Loop)>)
  - [钩子(Hook)](<#钩子(Hook)>)
    - [步骤1：创建一个新的钩子](#步骤1：创建一个新的钩子)
    - [步骤2：导入新的钩子](#步骤2：导入新的钩子)
    - [步骤3：修改配置文件](#步骤3：修改配置文件)
  - [优化器(Optimizer)](<#优化器(Optimizer)>)
    - [优化器包装器](#优化器包装器)
    - [构造器](#构造器)
  - [调度器(Scheduler)](<#调度器(Scheduler)>)

在本教程中，我们将介绍一些关于如何设置项目自定义运行的方法。

## 循环(Loop)

`Loop` 表示训练、验证或测试的工作流，我们使用 `train_cfg`,`val_cfg` 和 `test_cfg` 来构建 `Loop`。

示例:

```python
# Use EpochBasedTrainLoop to train 200 epochs.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
```

MMEngine定义了几个[基本的循环](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py)。如果定义的循环不满足需求，用户可以实现自定义的循环。

## 钩子(Hook)

在学习创建自定义钩子之前，建议先学习文件[engine.md](engine.md)中关于钩子的基本概念。

### 步骤1：创建一个新的钩子

根据钩子的目的，您需要根据期望的钩子点实现相应的函数。

例如，如果您想根据训练迭代次数和另外两个超参数的值在每个训练迭代后修改超参数的值，您可以实现一个类似以下的钩子：

```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook

from mmselfsup.registry import HOOKS
from mmselfsup.utils import get_model


@HOOKS.register_module()
class NewHook(Hook):
    """Docstring for NewHook.
    """

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        cur_iter = runner.iter
        get_model(runner.model).hyper_parameter = self.a * cur_iter + self.b
```

### 步骤2：导入新的钩子

然后我们需要确保 `NewHook` 已经被导入。假设 `NewHook` 在 `mmselfsup/engine/hooks/new_hook.py` 中，按照以下方式修改 `mmselfsup/engine/hooks/__init__.py` 文件：

```python
...
from .new_hook import NewHook

__all__ = [..., NewHook]
```

### 步骤3：修改配置文件

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value)
]
```

您还可以按照以下方式设置钩子的优先级：

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

默认情况下，在注册时，钩子的优先级被设置为 `NORMAL`。

## 优化器(Optimizer)

在自定义优化器配置之前，建议先学习文件[engine.md](engine.md)中有关优化器的基本概念。

以下是SGD优化器的示例：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

我们支持PyTorch中的所有优化器。更多细节，请参见[MMEngine优化器文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md)。

### 优化器包装器

优化器包装器提供了单精度训练和不同硬件的自动混合精度训练的统一接口。以下是一个 `optim_wrapper` 配置的示例：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

此外，如果您想要应用自动混合精度训练，可以修改上面的配置，例如：

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

`AmpOptimWrapper` 的 `loss_scale` 的默认设置为 `dynamic`。

### 构造器

构造器旨在构建优化器、优化器包装器并自定义不同层的超参数。配置文件中 `optim_wrapper` 的 `paramwise_cfg` 键控制此自定义。

有关示例和详细信息，请参见[MMEngine优化器文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md)。

此外，我们可以使用 `custom_keys` 为不同模块设置不同的超参数。

以下是MAE的 `optim_wrapper` 示例。以下配置将 `pos_embed`, `mask_token`, `cls_token` 模块和名称包含`ln`和`bias`的那些层的权重衰减乘法设置为0。在训练过程中，这些模块的权重衰减将是 `weight_decay * decay_mult`。

```python
optimizer = dict(
    type='AdamW', lr=1.5e-4 * 4096 / 256, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))
```

此外，对于某些特定设置，我们可以使用布尔类型的参数来控制优化过程或参数。例如，以下是SimCLR的示例配置：

```python
optimizer = dict(type='LARS', lr=0.3, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }))
```

在 `LARS` 优化器中，我们有 `lars_exclude` 选项来决定指定的层是否应用 `LARS` 优化方法。

## 调度器(Scheduler)

在自定义调度器配置之前，建议先学习 [MMEngine文档](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md) 中关于调度器的基本概念。

以下是一个调度器的示例：

```python
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]
```

**注意：** 当您更改 `train_cfg` 中的 `max_epochs` 时，请确保同时修改 `param_scheduler` 中的参数。
