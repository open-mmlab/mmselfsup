# Customize Runtime

- [Customize Runtime](#customize-runtime)
  - [Loop](#loop)
  - [Hook](#hook)
    - [Step 1: Create a new hook](#step-1-create-a-new-hook)
    - [Step 2: Import the new hook](#step-2-import-the-new-hook)
    - [Step 3: Modify the config](#step-3-modify-the-config)
  - [Optimizer](#optimizer)
    - [Optimizer Wrapper](#optimizer-wrapper)
    - [Constructor](#constructor)
  - [Scheduler](#scheduler)

In this tutorial, we will introduce some methods about how to customize runtime settings for the project.

## Loop

`Loop` means the workflow of training, validation or testing and we use `train_cfg`, `val_cfg` and `test_cfg` to build `Loop`.

E.g.:

```python
# Use EpochBasedTrainLoop to train 200 epochs.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
```

MMEngine defines several [basic loops](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py). Users could implement customized loops if the defined loops are not satisfied.

## Hook

Before learning to create your customized hooks, it is recommended to learn the basic concept of hooks in file [engine.md](engine.md).

### Step 1: Create a new hook

Depending on your intention of this hook, you need to implement corresponding functions according to the hook point of your expectation.

For example, if you want to modify the value of a hyper-parameter according to the training iter and two other hyper-parameters after every train iter, you could implement a hook like:

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

### Step 2: Import the new hook

Then we need to ensure `NewHook` imported. Assuming `NewHook` is in `mmselfsup/engine/hooks/new_hook.py`, modify `mmselfsup/engine/hooks/__init__.py` as below

```python
...
from .new_hook import NewHook

__all__ = [..., NewHook]
```

### Step 3: Modify the config

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook as below:

```python
custom_hooks = [
    dict(type='NewHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

By default, the hook's priority is set as `NORMAL` during registration.

## Optimizer

Before customizing the optimizer config, it is recommended to learn the basic concept of optimizer in file [engine.md](engine.md).

Here is an example of SGD optimizer:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
```

We support all optimizers of PyTorch. For more details, please refer to [MMEngine optimizer document](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

### Optimizer Wrapper

Optimizer wrapper provides a unified interface for single precision training and automatic mixed precision training with different hardware. Here is an example of `optim_wrapper` setting:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

Besides, if you want to apply automatic mixed precision training, you could modify the config above like:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

The default setting of `loss_scale` of `AmpOptimWrapper` is `dynamic`.

### Constructor

The constructor aims to build optimizer, optimizer wrapper and customize hyper-parameters of different layers. The key `paramwise_cfg` of `optim_wrapper` in configs controls this customization.

The example and detailed information can be found in [MMEngine optimizer document](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optim_wrapper.md).

Besides, We could use `custom_keys` to set different hyper-parameters of different modules.

Here is the `optim_wrapper` example of MAE. The config below sets weight decay multiplication to be 0 of `pos_embed`, `mask_token`, `cls_token` modules and those layers whose name contains `ln` and `bias`. During training, the weight decay of these modules will be `weight_decay * decay_mult`.

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

Furthermore, for some specific settings, we could use boolean type arguments to control the optimization process or parameters. For example, here is an example config of SimCLR:

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

In `LARS` optimizer, we have `lars_exclude` to decide whether the named layers apply the `LARS` optimization methods or not.

## Scheduler

Before customizing the scheduler config, it is recommended to learn the basic concept of scheduler in [MMEngine document](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md).

Here is an example of scheduler:

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

**Note:** When you change the `max_epochs` in `train_cfg`, make sure that the args in `param_scheduler` are modified simultanuously.
