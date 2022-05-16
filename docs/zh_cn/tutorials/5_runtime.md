# 教程 5：自定义模型运行参数

- [教程 5：自定义模型运行参数](#%E6%95%99%E7%A8%8B-5%EF%BC%9A%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B%E8%BF%90%E8%A1%8C%E5%8F%82%E6%95%B0)
  - [定制工作流](#%E5%AE%9A%E5%88%B6%E5%B7%A5%E4%BD%9C%E6%B5%81)
  - [钩子](#%E9%92%A9%E5%AD%90)
    - [默认训练钩子](#%E9%BB%98%E8%AE%A4%E8%AE%AD%E7%BB%83%E9%92%A9%E5%AD%90)
      - [权重文件钩子 CheckpointHook](#%E6%9D%83%E9%87%8D%E6%96%87%E4%BB%B6%E9%92%A9%E5%AD%90-checkpointhook)
      - [日志钩子 LoggerHooks](#%E6%97%A5%E5%BF%97%E9%92%A9%E5%AD%90-loggerhooks)
      - [验证钩子 EvalHook](#%E9%AA%8C%E8%AF%81%E9%92%A9%E5%AD%90-evalhook)
  - [使用其他内置钩子](#%E4%BD%BF%E7%94%A8%E5%85%B6%E4%BB%96%E5%86%85%E7%BD%AE%E9%92%A9%E5%AD%90)
  - [自定义钩子](#%E8%87%AA%E5%AE%9A%E4%B9%89%E9%92%A9%E5%AD%90)
    - [1. 创建一个新钩子](#1-%E5%88%9B%E5%BB%BA%E4%B8%80%E4%B8%AA%E6%96%B0%E9%92%A9%E5%AD%90)
    - [2. 导入新钩子](#2-%E5%AF%BC%E5%85%A5%E6%96%B0%E9%92%A9%E5%AD%90)
    - [3. 修改配置](#3-%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE)

在本教程中，我们将介绍如何在运行自定义模型时，进行自定义工作流和钩子的方法。

## 定制工作流

工作流是一个形如 (任务名，周期数) 的列表，用于指定运行顺序和周期。这里“周期数”的单位由执行器的类型来决定。

比如，我们默认使用基于**轮次**的执行器（`EpochBasedRunner`），那么“周期数”指的就是对应的任务在一个周期中要执行多少个轮次。通常，我们只希望执行训练任务，那么只需要使用以下设置：

```python
workflow = [('train', 1)]
```

有时我们可能希望在训练过程中穿插检查模型在验证集上的一些指标（例如，损失，准确率）。在这种情况下，可以将工作流程设置为：

```python
[('train', 1), ('val', 1)]
```

这样一来，程序会一轮训练一轮验证地反复执行。

默认情况下，我们更推荐在每个训练轮次后使用 **`EvalHook`** 进行模型验证。

## 钩子

钩子机制在 OpenMMLab 开源算法库中应用非常广泛，结合执行器可以实现对训练过程的整个生命周期进行管理，可以通过[相关文章](https://www.calltutors.com/blog/what-is-hook/)进一步理解钩子。

钩子只有被注册进执行器才起作用，目前钩子主要分为两类：

- 默认训练钩子

默认训练钩子由运行器默认注册，一般为一些基础型功能的钩子，已经有确定的优先级，一般不需要修改优先级。

- 定制钩子

定制钩子通过 `custom_hooks` 注册，一般为一些增强型功能的钩子，需要在配置文件中指定优先级，不指定该钩子的优先级将默被设定为 'NORMAL'。

**优先级列表**

|      Level      | Value |
| :-------------: | :---: |
|     HIGHEST     |   0   |
|    VERY_HIGH    |  10   |
|      HIGH       |  30   |
|  ABOVE_NORMAL   |  40   |
| NORMAL(default) |  50   |
|  BELOW_NORMAL   |  60   |
|       LOW       |  70   |
|    VERY_LOW     |  90   |
|     LOWEST      |  100  |

优先级确定钩子的执行顺序，每次训练前，日志会打印出各个阶段钩子的执行顺序，方便调试。

### 默认训练钩子

有一些常见的钩子未通过 `custom_hooks` 注册，但会在运行器（`Runner`）中默认注册，它们是：

|         Hooks         |     Priority      |
| :-------------------: | :---------------: |
|    `LrUpdaterHook`    |  VERY_HIGH (10)   |
| `MomentumUpdaterHook` |     HIGH (30)     |
|    `OptimizerHook`    | ABOVE_NORMAL (40) |
|   `CheckpointHook`    |    NORMAL (50)    |
|    `IterTimerHook`    |     LOW (70)      |
|      `EvalHook`       |     LOW (70)      |
|    `LoggerHook(s)`    |   VERY_LOW (90)   |

`OptimizerHook`，`MomentumUpdaterHook`和 `LrUpdaterHook` 在 [优化策略](./4_schedule.md) 部分进行了介绍， `IterTimerHook` 用于记录所用时间，目前不支持修改。

下面介绍如何使用去定制 `CheckpointHook`、`LoggerHooks` 以及 `EvalHook`。

#### 权重文件钩子 CheckpointHook

MMCV 的 runner 使用 `checkpoint_config` 来初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py)。

```python
checkpoint_config = dict(interval=1)
```

用户可以设置 `max_keep_ckpts` 来仅保存少量模型权重文件，或者通过 `save_optimizer`决定是否存储优化器的状态字典。更多细节可参考 [这里](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)。

#### 日志钩子 LoggerHooks

`log_config` 包装了多个记录器钩子，并可以设置间隔。
目前，MMCV 支持 `TextLoggerHook`、 `WandbLoggerHook`、`MlflowLoggerHook`、 `NeptuneLoggerHook`、 `DvcliveLoggerHook` 和 `TensorboardLoggerHook`。
更多细节可参考[这里](https://mmcv.readthedocs.io/zh_CN/latest/api.html#mmcv.runner.LoggerHook)。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 验证钩子 EvalHook

配置中的 `evaluation` 字段将用于初始化 [`EvalHook`](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/core/evaluation/eval_hooks.py).

`EvalHook` 有一些保留参数，如 `interval`，`save_best` 和 `start` 等。其他的参数，如 `metrics` 将被传递给 `dataset.evaluate()`。

```python
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})
```

我们可以通过参数 `save_best` 保存取得最好验证结果时的模型权重：

```python
# "auto" 表示自动选择指标来进行模型的比较。
# 也可以指定一个特定的 key 比如 "accuracy_top-1"。
evaluation = dict(interval=1, save_best="auto", metric='accuracy', metric_options={'topk': (1, )})
```

在跑一些大型实验时，可以通过修改参数 `start` 跳过训练靠前轮次时的验证步骤，以节约时间。如下：

```python
evaluation = dict(interval=1, start=200, metric='accuracy', metric_options={'topk': (1, )})
```

表示在第 200 轮之前，只执行训练流程，不执行验证；从轮次 200 开始，在每一轮训练之后进行验证。

## 使用其他内置钩子

一些钩子已在 MMCV 和 MMClassification 中实现：

- [EMAHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/ema.py)
- [SyncBuffersHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/sync_buffer.py)
- [EmptyCacheHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/memory.py)
- [ProfilerHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/profiler.py)
- ......

如果要用的钩子已经在MMCV中实现，可以直接修改配置以使用该钩子，如下格式：

```python
mmcv_hooks = [
    dict(type='MMCVHook', a=a_value, b=b_value, priority='NORMAL')
]
```

例如使用 `EMAHook`，进行一次 EMA 的间隔是100个 iter：

```python
custom_hooks = [
    dict(type='EMAHook', interval=100, priority='HIGH')
]
```

## 自定义钩子

### 1. 创建一个新钩子

这里举一个在 MMSelfSup 中创建一个新钩子的示例：

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

根据钩子的功能，用户需要指定钩子在训练的每个阶段将要执行的操作，比如 `before_run`，`after_run`，`before_epoch`，`after_epoch`，`before_iter` 和 `after_iter`。

### 2. 导入新钩子

之后，需要导入 `MyHook`。假设该文件在 `mmselfsup/core/hooks/my_hook.py`，有两种办法导入它：

- 修改 `mmselfsup/core/hooks/__init__.py` 进行导入，如下：

```python
from .my_hook import MyHook

__all__ = [..., MyHook, ...]
```

- 使用配置文件中的 `custom_imports` 变量手动导入

```python
custom_imports = dict(imports=['mmselfsup.core.hooks.my_hook'], allow_failed_imports=False)
```

### 3. 修改配置

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

还可通过 `priority` 参数设置钩子优先级，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

默认情况下，在注册过程中，钩子的优先级设置为 `NORMAL` 。
