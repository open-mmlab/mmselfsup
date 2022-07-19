# Tutorial 4: Customize Schedule

- [Tutorial 4: Customize Schedule](#tutorial-4-customize-schedule)
  - [Customize optimizer supported by Pytorch](#customize-optimizer-supported-by-pytorch)
  - [Customize learning rate schedules](#customize-learning-rate-schedules)
    - [Learning rate decay](#learning-rate-decay)
    - [Warmup strategy](#warmup-strategy)
    - [Customize momentum schedules](#customize-momentum-schedules)
    - [Parameter-wise configuration](#parameter-wise-configuration)
  - [Gradient clipping and gradient accumulation](#gradient-clipping-and-gradient-accumulation)
    - [Gradient clipping](#gradient-clipping)
    - [Gradient accumulation](#gradient-accumulation)
  - [Customize self-implemented optimizer](#customize-self-implemented-optimizer)

In this tutorial, we will introduce some methods about how to construct optimizers, customize learning rate, momentum schedules, parameter-wise configuration, gradient clipping, gradient accumulation, and customize self-implemented methods for the project.

## Customize optimizer supported by Pytorch

We already support to use all the optimizers implemented by PyTorch, and to use and modify them, please change the `optimizer` field of config files.

For example, if you want to use SGD, the modification could be as the following.

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, just modify the `lr` in the config of optimizer. You can also directly set other arguments according to the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

For example, if you want to use `Adam` with the setting like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch, the config should looks like:

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

In addition to optimizers implemented by PyTorch, we also implement a customized [LARS](https://arxiv.org/abs/1708.03888) in `mmselfsup/core/optimizer/optimizers.py`

## Customize learning rate schedules

### Learning rate decay

Learning rate decay is widely used to improve performance. And to use learning rate decay, please set the `lr_confg` field in config files.

For example, we use CosineAnnealing policy to train SimCLR, and the config is:

```python
lr_config = dict(
    policy='CosineAnnealing',
    ...)
```

Then during training, the program will call [CosineAnnealingLrUpdaterHook](https://github.com/open-mmlab/mmcv/blob/0092699fef27a0e6cbe9c37f4c4de2fb6996a1c7/mmcv/runner/hooks/lr_updater.py#L269) periodically to update the learning rate.

We also support many other learning rate schedules [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py), such as Poly schedule.

### Warmup strategy

In the early stage, training is easy to be volatile, and warmup is a technique to reduce volatility. With warmup, the learning rate will increase gradually from a small value to the expected value.

In MMSelfSup, we use `lr_config` to configure the warmup strategy, the main parameters are as follows：

- `warmup`: The warmup curve type. Please choose one from 'constant', 'linear', 'exp' and `None`, and `None` means disable warmup.
- `warmup_by_epoch` : whether warmup by epoch or not, default to be True, if set to be False, warmup by iter.
- `warmup_iters` : the number of warm-up iterations, when `warmup_by_epoch=True`, the unit is epoch; when `warmup_by_epoch=False`, the unit is the number of iterations (iter).
- `warmup_ratio` : warm-up initial learning rate will calculate as `lr = lr * warmup_ratio`.

Here are some examples:

1.linear & warmup by iter

```python
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20 * 1252,
    warmup_by_epoch=False)
```

2.exp & warmup by epoch

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='exp',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
```

### Customize momentum schedules

We support the momentum scheduler to modify the model's momentum according to learning rate, which could make the model converge in a faster way.

Momentum scheduler is usually used with LR scheduler, for example, the following config is used to accelerate convergence. For more details, please refer to the implementation of [CyclicLrUpdaterHook](https://github.com/open-mmlab/mmcv/blob/0092699fef27a0e6cbe9c37f4c4de2fb6996a1c7/mmcv/runner/hooks/lr_updater.py#L434) and [CyclicMomentumUpdaterHook](https://github.com/open-mmlab/mmcv/blob/0092699fef27a0e6cbe9c37f4c4de2fb6996a1c7/mmcv/runner/hooks/momentum_updater.py#L291).

Here is an example:

```python
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
```

### Parameter-wise configuration

Some models may have some parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layer and the bias in each layer. To finely configure them, we can use the `paramwise_options` in optimizer.

For example, if we do not want to apply weight decay to the parameters of BatchNorm or GroupNorm, and the bias in each layer, we can use following config file:

```python
optimizer = dict(
    type=...,
    lr=...,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0.),
        'bias': dict(weight_decay=0.)
    })
```

## Gradient clipping and gradient accumulation

### Gradient clipping

Besides the basic function of PyTorch optimizers, we also provide some enhancement functions, such as gradient clipping, gradient accumulation, etc. Please refer to [MMCV](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py) for more details.

Currently we support `grad_clip` option in `optimizer_config`, and you can refer to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) for more arguments .

Here is an example:

```python
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# norm_type: type of the used p-norm, here norm_type is 2.
```

When inheriting from base and modifying configs, if `grad_clip=None` in base, `_delete_=True` is needed.

### Gradient accumulation

When there is not enough computation resource, the batch size can only be set to a small value, which may degrade the performance of model. Gradient accumulation can be used to solve this problem.

Here is an example:

```python
data = dict(samples_per_gpu=64)
optimizer_config = dict(type="DistOptimizerHook", update_interval=4)
```

Indicates that during training, back-propagation is performed every 4 iters. And the above is equivalent to:

```python
data = dict(samples_per_gpu=256)
optimizer_config = dict(type="OptimizerHook")
```

## Customize self-implemented optimizer

In academic research and industrial practice, it is likely that you need some optimization methods not implemented by MMSelfSup, and you can add them through the following methods.

Implement your `CustomizedOptim` in `mmselfsup/core/optimizer/optimizers.py`

```python
import torch
from torch.optim import *  # noqa: F401,F403
from torch.optim.optimizer import Optimizer, required

from mmcv.runner.optimizer.builder import OPTIMIZERS

@OPTIMIZER.register_module()
class CustomizedOptim(Optimizer):

    def __init__(self, *args, **kwargs):

        ## TODO

    @torch.no_grad()
    def step(self):

        ## TODO
```

Import it in `mmselfsup/core/optimizer/__init__.py`

```python
from .optimizers import CustomizedOptim
from .builder import build_optimizer

__all__ = ['CustomizedOptim', 'build_optimizer', ...]
```

Use it in your config file

```python
optimizer = dict(
    type='CustomizedOptim',
    ...
)
```
