# 迁移文档

- [迁移文档](#迁移文档)
  - [迁移自 MMSelfSup 0.x 版本](#迁移自-mmselfsup-0x-版本)
  - [配置文件](#配置文件)
    - [数据集](#数据集)
    - [模型](#模型)
    - [优化器及调度](#优化器及调度)
    - [运行相关设置](#运行相关设置)
  - [代码包](#代码包)

## 迁移自 MMSelfSup 0.x 版本

我们将介绍一些 MMSelfSup 1.x 版本的变换，帮助用户更顺利的将项目从 MMSelfSup 0.x 版本迁移到 1.x 版本。

MMSelfSup 1.x 版本依赖于一些新的代码包，您应该根据 [安装教程](./get_started.md) 来创建新的环境，并安装依赖项。三个重要的依赖库已列出：

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine 是所有 OpenMMLab 2.0 项目的基础库，一部分非计算机视觉强相关的模块从 MMCV 迁移到了 MMEngine。
2. [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库。这不是新的依赖项，但是您需要将其升级到至少 `2.0.0rc1` 版本。
3. [MMClassification](https://github.com/open-mmlab/mmcv): OpenMMLab 图像分类代码库。这不是新的依赖项，但是您需要将其升级到至少 `1.0.0rc0` 版本。

## 配置文件

本章节将介绍 `_base_` 文件夹中的配置文件的变化，主要包含以下三个部分：

- 数据集：`mmselfsup/configs/selfsup/_base_/datasets`
- 模型：`mmselfsup/configs/selfsup/_base_/models`
- 优化器及调度：`mmselfsup/configs/selfsup/_base_/schedules`

### 数据集

在 **MMSelfSup 0.x** 中，我们使用字段 `data` 来整合数据相关信息, 例如 `samples_per_gpu`，`train`，`val` 等。

在 **MMSelfSup 1.x** 中，我们分别使用字段  `train_dataloader`, `val_dataloader` 整理训练和验证的数据相关信息，并且 `data` 字段已经被 **移除**。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
data = dict(
    samples_per_gpu=32,  # total 32*8(gpu)=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch,
    ),
    val=...)
```

</td>

<tr>
<td>新版本</td>
<td>

```python
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
val_dataloader = ...
```

</td>
</tr>
</table>

另外，我们 **移除** 了字段 `data_source`，以此来保证我们项目和其它 OpenMMLab 项目数据流的一致性。请查阅 [Config](user_guides/1_config.md) 获取更详细的信息。

**`pipeline`** 中的变化：

以 MAE 的 `pipeline` 作为例子，新的写法如下：

```python
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```

### 模型

在模型的配置文件中，和 MMSeflSup 0.x 版本相比，主要有两点不同。

1. 有一个新的字段 `data_preprocessor`，主要负责对数据进行预处理，例如归一化，通道转换等。例子如下：

```python
model = dict(
    type='MAE',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=...,
    neck=...,
    head=...,
    init_cfg=...)
```

**注意：** `data_preprocessor` 可以被定义在模型字段之外，外部定义将拥有更高优先级，并且覆盖模型字段内部定义。

例如以下写法中，`Runner` 将会基于 `mean=[123.675, 116.28, 103.53]` 和 `std=[58.395, 57.12, 57.375]` 这套参数进行构建 `data_preprocessor`，而忽略 `127.5` 的参数。

```python
data_preprocessor=dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True)
model = dict(
    type='MAE',
    data_preprocessor=dict(
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True)，
    backbone=...,
    neck=...,
    head=...,
    init_cfg=...)
```

相关 MMEngine 代码链接：[Runner](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L450) 会获取 `cfg.data_preprocessor`，并且 [合并](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L401) 进 `cfg.model`。

2. 在新版本的 `head` 字段中，我们新增加了 `loss`，主要负责损失函数的构建。例子如下：

```python
model = dict(
    type='MAE',
    data_preprocessor=...,
    backbone=...,
    neck=...,
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='MAEReconstructionLoss')),
    init_cfg=...)
```

### 优化器及调度

| MMSelfSup 0.x    | MMSelfSup 1.x   | 备注                                                                                                     |
| ---------------- | --------------- | -------------------------------------------------------------------------------------------------------- |
| optimizer_config | /               | `optimizer_config` 已经被**移除**。                                                                      |
| /                | optim_wrapper   | `optim_wrapper` 提供了参数更新的相关字段。                                                               |
| lr_config        | param_scheduler | `param_scheduler` 是一个列表设置学习率或者是其它参数，这将比之前更加灵活。                               |
| runner           | train_cfg       | `train_cfg` 中的循环设置（如 `EpochBasedTrainLoop`，`IterBasedTrainLoop`）将控制模型训练过程中的工作流。 |

1. **`optimizer`** 和 **`optimizer_config`** 的变化：

- 现在我们使用 `optim_wrapper` 字段来说明所有优化过程相关的设置，而
  `optimizer` 是 `optim_wrapper` 的一个子字段。
- `paramwise_cfg` 也是 `optim_wrapper` 的一个子字段，而不再是 `optimizer` 的子字段。
- `optimizer_config` 以经被**移除**，所有优化相关配置定义在 `optim_wrapper`中。
- `grad_clip` 重命名为 `clip_grad`。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
optimizer = dict(
    type='AdamW',
    lr=0.0015,
    weight_decay=0.3,
    paramwise_options = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ))
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
```

</td>
<tr>
<td>新版本</td>
<td>

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0015, weight_decay=0.3),
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_gard=dict(max_norm=1.0),
)
```

</td>
</tr>
</table>

2. **`lr_config`** 的变化：

- `lr_config` 已经被**移除**，并且我们使用新的字段 `param_scheduler` 来代替它。
- `warmup` 相关字段也被**移除**，因为我们使用学习率调度器组合来完成这项功能。新的调度器组合功能非常灵活，您可以用它设计各种不同的学习率或者动量变化曲线。参考 [教程](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html) 获取更多信息。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
```

</td>
<tr>
<td>新版本</td>
<td>

```python
param_scheduler = [
    # warmup
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=5,
        # Update the learning rate after every iters.
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5， end=200),
]
```

</td>
</tr>
</table>

3. **`runner`** 的变化:

在原来的 `runner` 字段中的配置已经被移到 `train_cfg`，`val_cfg` 和 `test_cfg`当中，主要控制训练、验证、测试等循环流程。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
runner = dict(type='EpochBasedRunner', max_epochs=200)
```

</td>
<tr>
<td>新版本</td>
<td>

```python
train_cfg = dict(by_epoch=True, max_epochs=200)
```

</td>
</tr>
</table>

### 运行相关设置

1. **`checkpoint_config`** 和 **`log_config`** 的变化：

`checkpoint_config` 相关配置被移动到了 `default_hooks.checkpoint` ，而 `log_config` 被移动到了 `default_hooks.logger`。

并且，我们将一些钩子相关的设置均移进 `default_hooks` 字段进行统一管理。

```python
default_hooks = dict(
    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch, and automatically save the best checkpoint.
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)
```

另外， 我们将原有的 `logger` 拆分为 `logger` 和 `visualizer`。`logger` 主要负责信息记录，而 `visualizer` 则是控制在不同后端来展示记录的信息，例如终端，TensorBoar 和 Wandb。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
```

</td>
<tr>
<td>新版本</td>
<td>

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=100),
)
visualizer = dict(
    type='SelfSupVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)
```

</td>
</tr>
</table>

2. **`load_from`** 和 **`resume_from`** 的变化：

- `resume_from` 已经被**移除**，我们使用 `resume` 和 `load_from` 代替它：

  - 如果 `resume=True` 并且 `load_from` 不是 None，将读取 `load_from` 字段的模型文件继续训练。
  - 如果 `resume=True` 并且 `load_from` 是 None，将在工作目录中尝试读取最近的模型文件继续训练。
  - 如果 `resume=False` 并且 `load_from` 不是 None，则只读取模型文件，不会继续训练。
  - 如果 `resume=False` 并且 `load_from` 是 None，不会读取模型文件，也不会继续训练，即随机初始化重新训练。

3. **`dist_params`** 的变化：

`dist_params` 字段现在是 `env_cfg` 的一部分，另外现在有一些新的配置在 `env_cfg` 当中。

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
```

4. **`workflow`** 的变化： `workflow` 相关字段已经被 **移除**.

5. 新字段 **`visualizer`**：

可视化器是 OpenMMLab 2.0 架构新设计的一部分。在 runner 中，我们使用可视化器的实例来处理结果和日志的可视化，并且将对应数据储存到不同的后端。
请查阅 [MMEngine 可视化文档](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html) 获取更多信息。

```python
visualizer = dict(
    type='SelfSupVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # Uncomment the below line to save the log and visualization results to TensorBoard.
        # dict(type='TensorboardVisBackend')
    ]
)
```

6. 新字段 **`default_scope`**： 起始点来搜索所有注册的模块。MMSelfSup 的`default_scope` 即是 `mmselfsup`。请查阅 [注册机制文档](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/registry.html) 获取更多信息

## 代码包

下列表格记录了代码模块、文件夹的主要改变。

| MMSelfSup 0.x         | MMSelfSup 1.x       | Remark                                                                                                                                               |
| --------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| apis                  | /                   | 目前 `apis` 文件夹已暂时被**移除**，在未来可能会再添加回来。                                                                                         |
| core                  | engine              | `core` 文件夹重命名为 `engine`，包含了 `hooks`，`opimizers`。                                                                                        |
| datasets              | datasets            | 数据集相关类主要基于不同的数据集实现，例如 ImageNet，Places205。                                                                                     |
| datasets/data_sources | /                   | `data_sources` 已经被**移除**，并且现在 `datasets` 的逻辑和 OpenMMLab 其它项目保持一致。                                                             |
| datasets/pipelines    | datasets/transforms | `pipelines` 文件夹已经重命名为 `transforms`。                                                                                                        |
| /                     | evaluation          | `evaluation` 主要负责管理一些评测函数或者是类，例如 KNN 等。                                                                                         |
| /                     | models/losses       | `losses` 文件夹提供了各种不同损失函数的实现。                                                                                                        |
| /                     | structures          | `structures` 文件夹提供了数据结构的实现。在 MMSelfSup 中，我们实现了一种新的数据结构，`selfsup_data_sample`，在训练/验证过程中来传输和接受数据信息。 |
| /                     | visualization       | `visualization` 文件夹包含了 visualizer，主要负责一些可视化的工作，例如数据增强的可视化等。                                                          |
