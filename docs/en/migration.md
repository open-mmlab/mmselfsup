# Migration

- [Migration](#migration)
  - [Migration from MMSelfSup 0.x](#migration-from-mmselfsup-0x)
  - [Config](#config)
    - [Datasets](#datasets)
    - [Models](#models)
    - [Schedules](#schedules)
    - [Runtime settings](#runtime-settings)
  - [Package](#package)

## Migration from MMSelfSup 0.x

We introduce some modifications of MMSelfSup 1.x, to help users to migrate their projects based on MMSelfSup 0.x to 1.x smoothly.

MMSelfSup 1.x depends on some new packages, you should create a new environment, with these required packages installed
according to the [install tutorial](./get_started.md). Three important packages are listed below,

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the base of all OpenMMLab 2.0 repos.
   Some modules, which are not specific to Computer Vision, are migrated from MMCV to this repo.
2. [MMCV](https://github.com/open-mmlab/mmcv): The computer vision package of OpenMMLab. This is not a new
   dependency, but you need to upgrade it to above `2.0.0rc1` version.
3. [MMClassification](https://github.com/open-mmlab/mmcv): The image classification package of OpenMMLab. This is not a new
   dependency, but you need to upgrade it to above `1.0.0rc0` version.

## Config

This section illustrates the changes of our config files in `_base_` folder, which includes three parts

- Datasets: `mmselfsup/configs/selfsup/_base_/datasets`
- Models: `mmselfsup/configs/selfsup/_base_/models`
- Schedules: `mmselfsup/configs/selfsup/_base_/schedules`

### Datasets

In **MMSelfSup 0.x**, we use key `data` to summarize all information, such as `samples_per_gpu`, `train`, `val`, etc.

In **MMSelfSup 1.x**, we separate `train_dataloader`, `val_dataloader` to summarize information correspodingly and the key `data` has been **removed**.

<table class="docutils">
<tr>
<td>Original</td>
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
<td>New</td>
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

Besides, we **remove** the key of `data_source` to keep the pipeline format consistent with that in other OpenMMLab projects. Please refer to [Config](user_guides/1_config.md) for more details.

Changes in **`pipeline`**:

Take MAE as an example of `pipeline`:

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

### Models

In the config of models, there are two main different parts from MMSeflSup 0.x.

1. There is a new key called `data_preprocessor`, which is responsible for preprocessing the data, like normalization, channel conversion, etc. For example:

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

**NOTE:** `data_preprocessor` can be defined outside the model dict, which has higher priority than it in model dict.

For example bwlow, `Runner` would build `data_preprocessor` based on `mean=[123.675, 116.28, 103.53]` and `std=[58.395, 57.12, 57.375]`, but omit the `127.5` of mean and std.

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

Related codes in MMEngine: [Runner](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L450) could get key `cfg.data_preprocessor` in `cfg` directly and [merge](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L401) it to `cfg.model`.

2. There is a new key `loss` in `head` in MMSelfSup 1.x, to determine the loss function of the algorithm. For example:

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

### Schedules

| MMSelfSup 0.x    | MMSelfSup 1.x   | Remark                                                                                                                          |
| ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| optimizer_config | /               | It has been **removed**.                                                                                                        |
| /                | optim_wrapper   | The `optim_wrapper` provides a common interface for updating parameters.                                                        |
| lr_config        | param_scheduler | The `param_scheduler` is a list to set learning rate or other parameters, which is more flexible.                               |
| runner           | train_cfg       | The loop setting (`EpochBasedTrainLoop`, `IterBasedTrainLoop`) in `train_cfg` controls the work flow of the algorithm training. |

1. Changes in **`optimizer`** and **`optimizer_config`**:

- Now we use `optim_wrapper` field to specify all configuration about the optimization process. And the
  `optimizer` is a sub field of `optim_wrapper` now.
- `paramwise_cfg` is also a sub field of `optim_wrapper`, instead of `optimizer`.
- `optimizer_config` is removed now, and all configurations of it are moved to `optim_wrapper`.
- `grad_clip` is renamed to `clip_grad`.

<table class="docutils">
<tr>
<td>Original</td>
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
<td>New</td>
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

2. Changes in **`lr_config`**:

- The `lr_config` field is removed and we use new `param_scheduler` to replace it.
- The `warmup` related arguments are removed, since we use a separate lr scheduler to implement this functionality. These introduced lr schedulers are very flexible, and you can use them to design many kinds of learning rate / momentum curves. See [the tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html) for more details.

<table class="docutils">
<tr>
<td>Original</td>
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
<td>New</td>
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

1. Changes in **`runner`**:

Most configuration in the original `runner` field is moved to `train_cfg`, `val_cfg` and `test_cfg`, which
configure the loop in training, validation and test.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
runner = dict(type='EpochBasedRunner', max_epochs=200)
```

</td>
<tr>
<td>New</td>
<td>

```python
train_cfg = dict(by_epoch=True, max_epochs=200)
```

</td>
</tr>
</table>

### Runtime settings

1. Changes in **`checkpoint_config`** and **`log_config`**:

The `checkpoint_config` are moved to `default_hooks.checkpoint` and the `log_config` are moved to `default_hooks.logger`.

And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.

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

In addition, we splited the original logger to logger and visualizer. The logger is used to record
information and the visualizer is used to show the logger in different backends, like terminal, TensorBoard
and Wandb.

<table class="docutils">
<tr>
<td>Original</td>
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
<td>New</td>
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

2. Changes in **`load_from`** and **`resume_from`**:

- The `resume_from` is removed. And we use `resume` and `load_from` to replace it.

  - If `resume=True` and `load_from` is not None, resume training from the checkpoint in `load_from`.
  - If `resume=True` and `load_from` is None, try to resume from the latest checkpoint in the work directory.
  - If `resume=False` and `load_from` is not None, only load the checkpoint, not resume training.
  - If `resume=False` and `load_from` is None, do not load nor resume.

3. Changes in **`dist_params`**:

The `dist_params` field is a sub field of `env_cfg` now. And there are some new configurations in the `env_cfg`.

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

4. Changes in **`workflow`**: `workflow` related functionalities are **removed**.

5. New field **`visualizer`**:

The visualizer is a new design in OpenMMLab 2.0 architecture. We use a
visualizer instance in the runner to handle results & log visualization and save to different backends.
See the [MMEngine visualization tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html) for more details.

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

1. New field **`default_scope`**: The start point to search module for all registries. The `default_scope` in MMSelfSup is `mmselfsup`. See [the registry tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/registry.html) for more details.

## Package

The table below records the general modification of the folders and files.

| MMSelfSup 0.x         | MMSelfSup 1.x       | Remark                                                                                                                                                                                                       |
| --------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| apis                  | /                   | Currently, the `apis` folder has been **removed**, it might be added in the future.                                                                                                                          |
| core                  | engine              | The `core` folder has been renamed to `engine`, which includes `hooks`, `opimizers`.                                                                                                                         |
| datasets              | datasets            | The datasets is implemented according to different datasets, such as ImageNet, Places205.                                                                                                                    |
| datasets/data_sources | /                   | The `data_sources` has been **removed** and the directory of `datasets` now is consistent with other OpenMMLab projects.                                                                                     |
| datasets/pipelines    | datasets/transforms | The `pipelines` folder has been renamed to `transforms`.                                                                                                                                                     |
| /                     | evaluation          | The `evaluation` is created for some evaluation functions or classes, such as KNN function or layer for detection.                                                                                           |
| /                     | models/losses       | The `losses` folder is created to provide different loss implementations, which is from `heads`                                                                                                              |
| /                     | structures          | The `structures` folder is for the implementation of data structures. In MMSelfSup, we implement a new data structure, `selfsup_data_sample`,  to pass and receive data throughout the training/val process. |
| /                     | visualization       | The `visualization` folder contains the visualizer, which is responsible for some visualization tasks like visualizing data augmentation.                                                                    |
