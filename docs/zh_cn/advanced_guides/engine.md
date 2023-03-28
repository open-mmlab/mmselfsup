# 训练引擎

<!-- TOC -->

- [训练引擎](#训练引擎)
  - [钩子(Hook)](#钩子hook)
    - [介绍](#介绍)
    - [默认钩子](#默认钩子)
    - [MMEngine中实现的常用钩子](#mmengine中实现的常用钩子)
    - [MMSelfsup 中实现的钩子](#mmselfsup-中实现的钩子)
  - [优化器](#优化器)
    - [优化器](#优化器-1)
      - [定制 PyTorch 支持的优化器](#定制-pytorch-支持的优化器)
      - [参数配置](#参数配置)
      - [MMSelfsup 中实现的优化器](#mmselfsup-中实现的优化器)
    - [优化器封装](#优化器封装)
      - [梯度裁剪](#梯度裁剪)
      - [梯度累加](#梯度累加)
      - [自动混合精度(AMP) 训练](#自动混合精度amp-训练)
    - [构造器](#构造器)
      - [MMSelfsup 中实现的构造器](#mmselfsup-中实现的构造器)

<!-- /TOC -->

## 钩子(Hook)

### 介绍

钩子机制在 OpenMMLab 开源算法库中被广泛使用,嵌入在`Runner`中,可以轻松管理训练过程的整个生命周期.您可以通过[相关文章](https://www.calltutors.com/blog/what-is-hook/)了解有关钩子的更多信息.

钩子只有在`Runner`注册后才能工作.目前钩子主要分为两类：

- 默认钩子 (default hooks)

这些钩子由`Runner`默认注册, 用以实现一些基本功能,并且具有默认的优先级,无需用户自行修改.

- 自定义钩子 (custom hooks)

自定义钩子通过`custom_hooks`注册.通常来讲,这些钩子主要用于功能增强,并且需要在配置文档中指定优先级.若没有指定钩子的优先级,则默认情况下会设置为`NORMAL`.

**优先级列表**:

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

优先级决定了钩子的执行顺序.在训练之前，日志会打印出每个阶段的钩子的执行顺序，方便调试.

### 默认钩子

以下常见的钩子由 MMEngine 中的[`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1750) 实现并在 [default](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/_base_/default_runtime.py#L3) 中进行了注册:

|                                                     Hooks                                                     |                                             Usage                                             |     Priority      |
| :-----------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: | :---------------: |
|    [RuntimeInfoHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/runtime_info_hook.py)    |                                   将运行时间更新到消息中心.                                   |  VERY_HIGH (10)   |
|      [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)      |                                   记录迭代过程所花费的时间.                                   |    NORMAL (50)    |
|  [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)  |                              确保分布式采样器 shuffle 是有效的.                               |    NORMAL (50)    |
|         [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)          | 从 `Runner` 的不同组件中收集日志记录, 并将其输出到终端, JSON 文件, tensorboard, wandb 等下游. | BELOW_NORMAL (60) |
| [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py) |                         更新优化器里面的一些超参数, 例如学习率和动量.                         |     LOW (70)      |
|     [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)      |                                     定期保存 checkpoint.                                      |   VERY_LOW (90)   |

### MMEngine中实现的常用钩子

在 MMEngine 中已经实现了一些钩子，它们是:

|                                                         Hooks                                                         |                                         Usage                                         |   Priority   |
| :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :----------: |
|                [EMAHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/ema_hook.py)                 |             在训练期间应用指数滑动平均 (Exponential Moving Average, EMA).             | NORMAL (50)  |
|         [EmptyCacheHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/empty_cache_hook.py)         |                       在训练过程中释放所有未占用的缓存GPU内存.                        | NORMAL (50)  |
|        [SyncBuffersHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sync_buffer_hook.py)         | 在每个Epoch结束时,在BN中同步模型缓冲区中的参数, 例如 `running_mean` 和 `running_var`. | NORMAL (50)  |
| [NaiveVisualizationHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/naive_visualization_hook.py) |                            在测试过程中显示或写出预测结果.                            | LOWEST (100) |

### MMSelfsup 中实现的钩子

在 MMSelfsup 中已经实现了一些钩子，它们是:

- [DeepClusterHook](mmselfsup.engine.hooks.DeepClusterHook)

- [DenseCLHook](mmselfsup.engine.hooks.DenseCLHook)

- [ODCHook](mmselfsup.engine.hooks.ODCHook)

- [SimSiamHook](mmselfsup.engine.hooks.SimSiamHook)

- [SwAVHook](mmselfsup.engine.hooks.SwAVHook)

- ......

例如:

以 [DenseCLHook](mmselfsup.engine.hooks.DenseCLHook) 为例, 这个钩子包含了 DenseCL 中的 `loss_lambda` 预热.

`loss_lambda` 是单一和稠密对比损失的损失权重. 默认值为 0.5.

```python
losses = dict()
losses['loss_single'] = loss_single * (1 - self.loss_lambda)
losses['loss_dense'] = loss_dense * self.loss_lambda
```

`DenseCLHook` 实现如下:

```python
...
@HOOKS.register_module()
class DenseCLHook(Hook):
...
    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
...
        cur_iter = runner.iter
        if cur_iter >= self.start_iters:
            get_model(runner.model).loss_lambda = self.loss_lambda
        else:
            get_model(runner.model).loss_lambda = 0.

```

若钩子已在 MMEngine 或 MMSelfsup 中实现，则可以直接修改配置来使用这个钩子，如下所示:

```python
custom_hooks = [
    dict(type='MMEngineHook', a=a_value, b=b_value, priority='NORMAL')
]
```

例如使用  `DenseCLHook`, start_iters 是 500:

```python
custom_hooks = [
    dict(type='DenseCLHook', start_iters=500)
]
```

## 优化器

下面将通过 3 个不同的部分来介绍优化器章节: 优化器、优化器封装和构造器

### 优化器

#### 定制 PyTorch 支持的优化器

我们已经支持了 PyTorch 实现的所有优化器，可参阅 `mmengine/optim/optimizer/builder.py`. 若要使用或修改,请更改配置文件的 `optimizer` 字段.

例如, 若要使用 SGD,则可进行如下修改.

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

要修改模型的学习率，只需优化器配置中修改 `lr` . 同时也可以根据 PyTorch 的 [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 直接设置其他参数 .

例如,如果您期望使用如 PyTorch 中 `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` 的 `Adam` 设置, 配置应该看起来像:

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

#### 参数配置

有些模型在优化时可能有一些特定的参数设置, 例如不需要将权重衰减应用到 `BatchNorm` 层和每一层的`bias`中. 为了精确地配置它们, 我们可以使用优化器中的 `paramwise_cfg`.

例如, 在 MAE 中, 我们不想对 `ln`, `bias`, `pos_embed`, `mask_token` 和 `cls_token` 等参数应用权重衰减, 我们可以使用以下配置文件:

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

#### MMSelfsup 中实现的优化器

- [LARS](mmselfsup.engine.optimizers.LARS)

除了PyTorch实现的优化器之外, 我们还在`mmselfsup/engine/optimizers/lars.py` 中实现了一个定制的 [LARS](mmselfsup.engine.optimizers.LARS),为SGD实现了分层自适应学习率缩放.

```python
optimizer = dict(type='LARS', lr=4.8, momentum=0.9, weight_decay=1e-6)
```

### 优化器封装

除了PyTorch优化器的基本功能外, 我们还提供了一些增强功能, 例如 梯度裁剪, 梯度累加, 自动混合精度训练等. 更多细节请参考 [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py).

#### 梯度裁剪

目前我们在`optim_wrapper`中支持`clip_grad`选项，您可以参考 [OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17) 和 [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) 文档了解更多的参数.下面是一个例子:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper',
	optimizer=optimizer,
    clip_grad=dict(
        max_norm=0.2,
        norm_type=2))
# norm_type: type of the used p-norm, here norm_type is 2.
```

如果 `clip_grad` 不是 `None`, 它将是 `torch.nn.utils.clip_grad.clip_grad_norm_()` 的参数.

#### 梯度累加

当没有足够的计算资源时,批量大小只能设置为一个小批量,这可能会降低模型的性能. 梯度累加可以用来解决这个问题.

下面是一个例子:

```python
train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=4)
```

这个例子表示在训练期间,每4个iter进行一次反向传播. 并且上述内容相当于:

```python
train_dataloader = dict(batch_size=256)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=1)
```

#### 自动混合精度(AMP) 训练

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

`AmpOptimWrapper` 中 `loss_scale` 的默认设置是 `dynamic`.

### 构造器

构造器旨在建立优化器、优化器封装以及定制不同层的超参数.配置文件中 `optim_wrapper` 的 `paramwise_cfg` 函数控制这种定制.

#### MMSelfsup 中实现的构造器

- [LearningRateDecayOptimWrapperConstructor](mmselfsup.engine.optimizers.LearningRateDecayOptimWrapperConstructor)

`LearningRateDecayOptimWrapperConstructor` 为主干网络的不同层设置不同的学习率. 注意: 目前,这个优化器构造器是为 ViT, Swin, MixMIM 构建的.

一个例子:

```python
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-3, model_type='swin', layer_decay_rate=0.9),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor')
```

注意: `paramwise_cfg` 只支持 `weight_decay` 的自定义.
