# 教程 4: 使用自定义数据集进行预训练

- [教程 4: 使用自定义数据集进行预训练](#教程-4-使用自定义数据集进行预训练)
  - [在自定义数据集上使用MAE算法进行预训练](#在自定义数据集上使用mae算法进行预训练)
    - [第一步：获取自定义数据路径](#第一步获取自定义数据路径)
    - [第二步：选择一个配置文件作为模板](#第二步选择一个配置文件作为模板)
    - [第三步：修改数据集相关的配置](#第三步修改数据集相关的配置)
  - [在COCO数据集上使用MAE算法进行预训练](#在coco数据集上使用mae算法进行预训练)
  - [在自定义数据集上使用SimCLR算法进行预训练](#在自定义数据集上使用simclr算法进行预训练)
  - [使用MMSelfSup提供的预训练模型来加速收敛](#使用mmselfsup提供的预训练模型来加速收敛)

在本教程中，我们将介绍如何使用自定义数据集(无需标注)进行自监督预训练。

## 在自定义数据集上使用 MAE 算法进行预训练

在MMSelfSup中, 我们支持用户直接调用MMClassification的`CustomDataset`(类似于`torchvision`的`ImageFolder`), 该数据集能自动的读取给的路径下的图片。你只需要准备你的数据集路径，并修改配置文件，即可轻松使用MMSelfSup进行预训练。

### 第一步：获取自定义数据路径

路径应类似这种形式： `data/custom_dataset/`

### 第二步：选择一个配置文件作为模板

在本教程中，我们使用 `configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py`作为一个示例进行讲解。我们首先复制这个配置文件，将新复制的文件命名为`mae_vit-base-p16_8xb512-coslr-400e_${custom_dataset}.py`.

- `custom_dataset`: 表明你用的那个数据集。例如，用 `in1k` 代表ImageNet 数据集，`coco` 代表COCO数据集。

这个配置文件的内容如下：

```python
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset 8 x 512
train_dataloader = dict(batch_size=512, num_workers=8)

# optimizer wrapper
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

# learning rate scheduler
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

# runtime settings
# pre-train for 400 epochs
train_cfg = dict(max_epochs=400)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True
```

### 第三步：修改数据集相关的配置

数据集相关的配置是定义在 `_base_`的`'../_base_/datasets/imagenet_mae.py'` 文件内。我们直接将其内容复制到刚刚创建的新的配置文件 `mae_vit-base-p16_8xb512-coslr-400e_${custom_dataset}.py` 中.

- 此时我们删除 `_base_`的 `'../_base_/datasets/imagenet_mae.py'`。
- 修改`dataset_type = 'mmcls.CustomDataset'`和` data_root = /dataset/my_custom_dataset`.
- 删除 `train_dataloader`中的 `ann_file` ，同时根据自己的实际情况决定是否需要设定 `data_prefix`。

```{note}
`CustomDataset` 是在MMClassification实现的, 因此我们使用这种方式 `dataset_type=mmcls.CustomDataset` 来使用这个类。
```

此时，修改后的文件应如下：

```python
# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    # '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'mmcls.CustomDataset'

data_root = 'data/custom_dataset/'
file_client_args = dict(backend='disk')
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

# dataset 8 x 512
train_dataloader = dict(
    batch_size=512,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='meta/train.txt', # removed if you don't have the annotation file
        data_prefix=dict(img_path='./'),
        pipeline=train_pipeline))
# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<


# optimizer wrapper
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

# learning rate scheduler
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

# runtime settings
# pre-train for 400 epochs
train_cfg = dict(max_epochs=400)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True
```

使用上述配置文件，你就能够轻松的在自定义数据集上使用MAE算法来进行预训练了。

## 在COCO数据集上使用MAE算法进行预训练

```{note}
你可能需要参考[文档](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/get_started.md)安装MMDetection 来使用 `mmdet.CocoDataset`。
```

与在自定义数据集上进行预训练类似，我们在本教程中也提供了一个使用COCO数据集进行预训练的示例。修改后的文件如下：

```python
# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    # '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')

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

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))

# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<

# optimizer wrapper
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

# learning rate scheduler
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

# runtime settings
# pre-train for 400 epochs
train_cfg = dict(max_epochs=400)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True
```

## 在自定义数据集上使用SimCLR算法进行预训练

我们也提供了一个使用SimCLR在自定义数据集上进行预训练的配置文件，主要思路与 [在自定义数据集上使用MAE算法进行预训练](#在自定义数据集上使用mae算法进行预训练) 是类似的。

我们使用的模板是 `configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py`，你可以根据自己的需要从配置文件仓库里选择合适的文件作为模板，其修改后的内容如下:

```python
# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/simclr.py',
    # '../_base_/datasets/imagenet_simclr.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'mmcls.CustomDataset'
data_root = 'data/custom_dataset/'
file_client_args = dict(backend='disk')

view_pipeline = [
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='meta/train.txt',
        data_prefix=dict(img_path='./'),
        pipeline=train_pipeline))
# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<


# optimizer
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

# runtime settings
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

```

## 使用MMSelfSup提供的预训练模型来加速收敛

在具体应用中，我们可以使用MMSelfSup已经提供的预训练模型来加速自定义数据集上的训练速度。你可以考虑使用这些预训练模型作为初始化。具体来讲，你只需要从 [模型库](https://mmselfsup.readthedocs.io/en/1.x/model_zoo.html) 中选择一个合适模型，获取模型权重的URL链接，并在启动训练的时候，指定这个链接作为预训练模型。

```bash
bash tools/dist_train.sh ${CONFIG} ${GPUS} --cfg-options model.pretrained=${PRETRAIN}
```

- `CONFIG`: 修改后的配置文件
- `GPUS`: 使用的GPU数
- `PRETRAIN`: MMSelfSup提供的预训练模型文件的URL
