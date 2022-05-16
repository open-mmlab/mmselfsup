# 教程 4：自定义优化策略

- [教程 4：自定义优化策略](#%E6%95%99%E7%A8%8B-4%EF%BC%9A%E8%87%AA%E5%AE%9A%E4%B9%89%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5)
  - [构造 PyTorch 内置优化器](#%E6%9E%84%E9%80%A0-pytorch-%E5%86%85%E7%BD%AE%E4%BC%98%E5%8C%96%E5%99%A8)
  - [定制学习率调整策略](#%E5%AE%9A%E5%88%B6%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5)
    - [定制学习率衰减曲线](#%E5%AE%9A%E5%88%B6%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%A1%B0%E5%87%8F%E6%9B%B2%E7%BA%BF)
    - [定制学习率预热策略](#%E5%AE%9A%E5%88%B6%E5%AD%A6%E4%B9%A0%E7%8E%87%E9%A2%84%E7%83%AD%E7%AD%96%E7%95%A5)
    - [定制动量调整策略](#%E5%AE%9A%E5%88%B6%E5%8A%A8%E9%87%8F%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5)
    - [参数化精细配置](#%E5%8F%82%E6%95%B0%E5%8C%96%E7%B2%BE%E7%BB%86%E9%85%8D%E7%BD%AE)
  - [梯度裁剪与梯度累计](#%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%B4%AF%E8%AE%A1)
    - [梯度裁剪](#%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA)
    - [梯度累计](#%E6%A2%AF%E5%BA%A6%E7%B4%AF%E8%AE%A1)
  - [用户自定义优化方法](#%E7%94%A8%E6%88%B7%E8%87%AA%E5%AE%9A%E4%B9%89%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95)

在本教程中，我们将介绍如何在运行自定义模型时，进行构造优化器、定制学习率、动量调整策略、参数化精细配置、梯度裁剪、梯度累计以及用户自定义优化方法等。

## 构造 PyTorch 内置优化器

我们已经支持使用PyTorch实现的所有优化器，要使用和修改这些优化器，请修改配置文件中的`optimizer`字段。

例如，如果您想使用SGD，可以进行如下修改。

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

要修改模型的学习率，只需要在优化器的配置中修改 `lr` 即可。 要配置其他参数，可直接根据 [PyTorch API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim)进行。

例如，如果想使用 `Adam` 并设置参数为 `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)`， 则需要进行如下配置

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

除了PyTorch实现的优化器之外，我们还在 `mmselfsup/core/optimizer/optimizers.py` 中构造了一个[LARS](https://arxiv.org/abs/1708.03888)。

## 定制学习率调整策略

### 定制学习率衰减曲线

深度学习研究中，广泛应用学习率衰减来提高网络的性能。要使用学习率衰减，可以在配置中设置 `lr_confg` 字段。

例如，在 SimCLR 网络训练中，我们使用 CosineAnnealing 的学习率衰减策略，配置文件为：

```python
lr_config = dict(
    policy='CosineAnnealing',
    ...)
```

在训练过程中，程序会周期性地调用 MMCV 中的 [CosineAnealingLrUpdaterHook](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L227) 来进行学习率更新。

此外，我们也支持其他学习率调整方法，如 `Poly` 等。详情可见 [这里](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py)

### 定制学习率预热策略

在训练的早期阶段，网络容易不稳定，而学习率的预热就是为了减少这种不稳定性。通过预热，学习率将会从一个很小的值逐步提高到预定值。

在 MMSelfSup 中，我们同样使用 `lr_config` 配置学习率预热策略，主要的参数有以下几个：

- `warmup` : 学习率预热曲线类别，必须为 'constant'、 'linear'， 'exp' 或者 `None` 其一， 如果为 `None`, 则不使用学习率预热策略。
- `warmup_by_epoch` : 是否以轮次（epoch）为单位进行预热，默认为 True 。如果被设置为 False ， 则以 iter 为单位进行预热。
- `warmup_iters` : 预热的迭代次数，当 `warmup_by_epoch=True` 时，单位为轮次（epoch）；当 `warmup_by_epoch=False` 时，单位为迭代次数（iter）。
- `warmup_ratio` : 预热的初始学习率 `lr = lr * warmup_ratio`。

例如：

1.逐**迭代次数**地**线性**预热

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

2.逐**轮次**地**指数**预热

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='exp',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
```

### 定制动量调整策略

我们支持动量调整器根据学习率修改模型的动量，从而使模型收敛更快。

动量调整策略通常与学习率调整策略一起使用，例如，以下配置用于加速收敛。更多细节可参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327) 和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130)。

例如：

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

### 参数化精细配置

一些模型的优化策略，包含作用于特定参数的精细设置，例如 BatchNorm 层不添加权重衰减或者对不同的网络层使用不同的学习率。为了进行精细配置，我们通过 `optimizer` 中的 `paramwise_options` 参数进行配置。

例如，如果我们不想对 BatchNorm 或 GroupNorm 的参数以及各层的 bias 应用权重衰减，我们可以使用以下配置文件：

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

## 梯度裁剪与梯度累计

### 梯度裁剪

除了 PyTorch 优化器的基本功能，我们还提供了一些增强功能，例如梯度裁剪、梯度累计等。更多细节参考 [MMCV](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py)。

目前我们支持在 `optimizer_config` 字段中添加 `grad_clip` 参数来进行梯度裁剪，更详细的参数可参考 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)。

用例如下：

```python
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# norm_type: 使用的范数类型，此处使用范数2。
```

当使用继承并修改基础配置时，如果基础配置中 `grad_clip=None`，需要添加 `_delete_=True`。

### 梯度累计

计算资源缺乏时，每个批次的大小（batch size）只能设置为较小的值，这可能会影响模型的性能。可以使用梯度累计来规避这一问题。

用例如下：

```python
data = dict(samples_per_gpu=64)
optimizer_config = dict(type="DistOptimizerHook", update_interval=4)
```

表示训练时，每 4 个 iter 执行一次反向传播。由于此时单张 GPU 上的批次大小为 64，也就等价于单张 GPU 上一次迭代的批次大小为 256，也即：

```python
data = dict(samples_per_gpu=256)
optimizer_config = dict(type="OptimizerHook")
```

## 用户自定义优化方法

在学术研究和工业实践中，可能需要使用 MMSelfSup 未实现的优化方法，可以通过以下方法添加。

在 `mmselfsup/core/optimizer/optimizers.py` 中实现您的 `CustomizedOptim` 。

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

修改 `mmselfsup/core/optimizer/__init__.py`，将其导入

```python
from .optimizers import CustomizedOptim
from .builder import build_optimizer

__all__ = ['CustomizedOptim', 'build_optimizer', ...]
```

在配置文件中指定优化器

```python
optimizer = dict(
    type='CustomizedOptim',
    ...
)
```
