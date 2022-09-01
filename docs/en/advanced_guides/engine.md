# Engine

<!-- TOC -->

- [Engine](#engine)
  - [Hook](#hook)
    - [Introduction](#introduction)
    - [Default hooks](#default-hooks)
    - [Common Hooks implemented in MMEngine](#common-hooks-implemented-in-mmengine)
    - [Hooks implemented in MMSelfsup](#hooks-implemented-in-mmselfsup)
  - [Optimizer](#optimizer)
    - [Optimizer](#optimizer-1)
      - [Customize optimizer supported by PyTorch](#customize-optimizer-supported-by-pytorch)
      - [Parameter-wise configuration](#parameter-wise-configuration)
      - [Implemented optimizers in MMSelfsup](#implemented-optimizers-in-mmselfsup)
    - [Optimizer wrapper](#optimizer-wrapper)
      - [Gradient clipping](#gradient-clipping)
      - [Gradient accumulation](#gradient-accumulation)
      - [Automatic mixed precision(AMP) training](#automatic-mixed-precisionamp-training)
    - [Constructor](#constructor)
      - [Constructors implemented in MMSelfsup](#constructors-implemented-in-mmselfsup)

<!-- /TOC -->

## Hook

### Introduction

The hook mechanism is widely used in the OpenMMLab open-source algorithm library. Inserted in the `Runner`, the entire life cycle of the training process can be managed easily. You can learn more about the hook through [related article](https://www.calltutors.com/blog/what-is-hook/).

Hooks only work after being registered into the runner. At present, hooks are mainly divided into two categories:

- default hooks

Those hooks are registered by the runner by default. Generally, they fulfill some basic functions, and have default priority, you don't need to modify the priority.

- custom hooks

The custom hooks are registered through custom_hooks. Generally, they are hooks with enhanced functions. The priority needs to be specified in the configuration file. If you do not specify the priority of the hook, it will be set to 'NORMAL' by default.

**Priority list**:

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

The priority determines the execution order of the hooks. Before training, the log will print out the execution order of the hooks at each stage to facilitate debugging.

### Default hooks

The following common hooks are already reigistered by [default](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/_base_/default_runtime.py#L3), which is implemented through [`register_default_hooks`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L1750) in MMEngine:

|                                                     Hooks                                                     |                                                         Usage                                                         |     Priority      |
| :-----------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :---------------: |
|    [RuntimeInfoHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/runtime_info_hook.py)    |                                     update runtime information into message hub.                                      |  VERY_HIGH (10)   |
|      [IterTimerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/iter_timer_hook.py)      |                                         log the time spent during iteration.                                          |    NORMAL (50)    |
|  [DistSamplerSeedHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sampler_seed_hook.py)  |                                     ensure distributed Sampler shuffle is active                                      |    NORMAL (50)    |
|         [LoggerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py)          | collect logs from different components of `Runner` and write them to terminal, JSON file, tensorboard and wandb .etc. | BELOW_NORMAL (60) |
| [ParamSchedulerHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/param_scheduler_hook.py) |                     update some hyper-parameters in optimizer, e.g., learning rate and momentum.                      |     LOW (70)      |
|     [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py)      |                                            save checkpoints periodically.                                             |   VERY_LOW (90)   |

### Common Hooks implemented in MMEngine

Some hooks have been already implemented in MMEngine, they are:

|                                                         Hooks                                                         |                                             Usage                                              |   Priority   |
| :-------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :----------: |
|                [EMAHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/ema_hook.py)                 |              apply Exponential Moving Average (EMA) on the model during training.              | NORMAL (50)  |
|         [EmptyCacheHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/empty_cache_hook.py)         |            release all unoccupied cached GPU memory during the process of training.            | NORMAL (50)  |
|        [SyncBuffersHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/sync_buffer_hook.py)         | synchronize model buffers such as running_mean and running_var in BN at the end of each epoch. | NORMAL (50)  |
| [NaiveVisualizationHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/naive_visualization_hook.py) |               Show or Write the predicted results during the process of testing.               | LOWEST (100) |

### Hooks implemented in MMSelfsup

Some hooks have been already implemented in MMSelfsup, they are:

- [DeepClusterHook](mmselfsup.engine.hooks.DeepClusterHook)

- [DenseCLHook](mmselfsup.engine.hooks.DenseCLHook)

- [ODCHook](mmselfsup.engine.hooks.ODCHook)

- [SimSiamHook](mmselfsup.engine.hooks.SimSiamHook)

- [SwAVHook](mmselfsup.engine.hooks.SwAVHook)

- ......

An example:

Take [DenseCLHook](mmselfsup.engine.hooks.DenseCLHook) for example, this hook includes `loss_lambda` warmup in DenseCL.

`loss_lambda` is loss weight for the single and dense contrastive loss. Defaults to 0.5.

```python
losses = dict()
losses['loss_single'] = loss_single * (1 - self.loss_lambda)
losses['loss_dense'] = loss_dense * self.loss_lambda
```

`DenseCLHook` is implemented as follows:

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

If the hook is already implemented in MMEngine or MMSelfsup, you can directly modify the config to use the hook as below

```python
custom_hooks = [
    dict(type='MMEngineHook', a=a_value, b=b_value, priority='NORMAL')
]
```

such as using `EMAHook`, start_iters is 500:

```python
custom_hooks = [
    dict(type='DenseCLHook', start_iters=500)
]
```

## Optimizer

We will introduce Optimizer section through 3 different parts: Optimizer, Optimizer wrapper, and Constructor.

### Optimizer

#### Customize optimizer supported by PyTorch

We have already supported all the optimizers implemented by PyTorch, see `mmengine/optim/optimizer/builder.py`. To use and modify them, please change the `optimizer` field of config files.

For example, if you want to use SGD, the modification could be as the following.

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, just modify the `lr` in the config of optimizer. You can also directly set other arguments according to the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

For example, if you want to use `Adam` with the setting like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch, the config should looks like:

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

#### Parameter-wise configuration

Some models may have some parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layer and the bias in each layer. To finely configure them, we can use the `paramwise_cfg` in optimizer.

For example, in MAE, we do not want to apply weight decay to the parameters of `ln`, `bias`, `pos_embed`, `mask_token` and `cls_token`, so we can use following config file:

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

#### Implemented optimizers in MMSelfsup

- [LARS](mmselfsup.engine.optimizers.LARS)

In addition to optimizers implemented by PyTorch, we also implement a customized [LARS](mmselfsup.engine.optimizers.LARS) in `mmselfsup/engine/optimizers/lars.py`. It implements layer-wise adaptive rate scaling for SGD.

```python
optimizer = dict(type='LARS', lr=4.8, momentum=0.9, weight_decay=1e-6)
```

### Optimizer wrapper

Besides the basic function of PyTorch optimizers, we also provide some enhancement functions, such as gradient clipping, gradient accumulation, automatic mixed precision training, etc. Please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py) for more details.

#### Gradient clipping

Currently we support `clip_grad` option in `optim_wrapper`, and you can refer to [OptimWrapper](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17) and [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)for more arguments . Here is an example:

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

If `clip_grad` is not None, it will be the arguments of `torch.nn.utils.clip_grad.clip_grad_norm_()`.

#### Gradient accumulation

When there is not enough computation resource, the batch size can only be set to a small value, which may degrade the performance of model. Gradient accumulation can be used to solve this problem.

Here is an example:

```python
train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=4)
```

Indicates that during training, back-propagation is performed every 4 iters. And the above is equivalent to:

```python
train_dataloader = dict(batch_size=256)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    accumulative_counts=1)
```

#### Automatic mixed precision(AMP) training

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

The default setting of `loss_scale` of `AmpOptimWrapper` is `dynamic`.

### Constructor

The constructor aims to build optimizer, optimizer wrapper and customize hyper-parameters of different layers. The key `paramwise_cfg` of `optim_wrapper` in configs controls this customization.

#### Constructors implemented in MMSelfsup

- [LearningRateDecayOptimWrapperConstructor](mmselfsup.engine.optimizers.LearningRateDecayOptimWrapperConstructor)

`LearningRateDecayOptimWrapperConstructor` sets different learning rates for different layers of backbone. Note: Currently, this optimizer constructor is built for ViT and Swin.

An example:

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

Note: `paramwise_cfg` will be ignored, and it can be written as  `paramwise_cfg=dict()` .  By default, `LearningRateDecayOptimWrapperConstructor` will not apply weight decay to `normalization parameters`, `bias`, `position embedding`, `class token`, and `relative position bias table`, automatically.
