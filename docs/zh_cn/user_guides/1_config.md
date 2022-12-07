# 教程 1: 了解配置文件

- [教程 1: 了解配置文件](#教程-1-了解配置文件)
  - [配置文件命名规则](#配置文件命名规则)
    - [算法信息](#算法信息)
    - [模块信息](#模块信息)
    - [训练信息](#训练信息)
    - [数据信息](#数据信息)
    - [配置文件命名示例](#配置文件命名示例)
  - [配置文件结构](#配置文件结构)
  - [继承和修改配置文件](#继承和修改配置文件)
    - [使用配置中的中间变量](#使用配置中的中间变量)
    - [忽略基础配置中的字段](#忽略基础配置中的字段)
    - [使用基础配置中的字段](#使用基础配置中的字段)
  - [通过脚本参数修改配置](#通过脚本参数修改配置)
  - [导入用户定义模块](#导入用户定义模块)

MMSelfSup 主要是在 python 文件中来设置各种各样的配置。我们配置文件系统的设计融合了模块化和可继承的设计理念，可以让用户轻松方便地完成各种实验配置。所有的配置文件全部位于 `configs` 目录下。如果您想查看配置文件的全貌，您可以使用以下命令 `python tools/misc/print_config.py`。

## 配置文件命名规则

我们使用以下规则来命名我们的配置文件，社区贡献者建议遵循这个规则来贡献您的代码。简单来说，配置文件的名字主要划分为四个部分: `algorithm info`, `module information`, `training information` 和 `data information`。不同部分通过下划线 `_` 来进行相连，而属于同一个部分的内容，通过中横线 `-`来进行相连。

我们使用以下一个实例让大家有一个清晰的认识

```
{algorithm_info}_{module_info}_{training_info}_{data_info}.py
```

- `algorithm_info`：与算法相关的一些信息，例如算法名;
- `module_info`： 模块相关的一些信息，例如与loss, head相关的信息;
- `training_info`：训练相关的信息, 例如 batch size, 学习率调整器和数据增强策略。
- `data_info`：数据相关信息, 例如数据集名，输入图片的大小;

在下面几个章节，我们将对文件名中的各个部分进行详细的说明：

### 算法信息

```
{algorithm}-{misc}
```

`algorithm` 通常情况下是算法名字的缩写和版本号. 例如:

- `relative-loc` : 算法名中不同的部分通过中横线 `-`相连
- `simclr`
- `mocov2`

`misc` 描述了算法的一些其他信息

- `npid-ensure-neg`
- `deepcluster-sobel`

### 模块信息

```
{backbone_setting}-{neck_setting}-{head_setting}-{loss_setting}
```

模块信息大部分情况下是有关 backbone 的一些信息. 例如:

- `resnet50`
- `vit-base-p16`
- `swin-base`

有时候，有些特殊的配置需要在配置文件名中提及，例如:

- `resnet50-sobel`: 在诸如线性评测之类的下游任务, 当我们使用的是 DeepCluster 的预训练模型，在经过 Sobel 层之后，模型只接受两层输入

而 `neck_setting`, `head_setting` 和 `loss_setting` 这几个选项是可选的。

### 训练信息

训练相关的一些配置，包括 batch size, 学习率调整方案和数据增强等。

- Batch size, 其格式为 `{gpu x batch_per_gpu}`，如 `8xb32`;
- 训练配置, 他们需要以下面这个格式来进行书写 `{pipeline aug}-{train aug}-{scheduler}-{epochs}`

如：

- `8xb32-mcrop-2-6-coslr-200e` : `mcrop` 是 SwAV 提出的 pipeline 中的名为 multi-crop 的一部分。2 和 6 表示 2 个 pipeline 分别输出 2 个和 6 个裁剪图，而且裁剪信息记录在数据信息中；
- `8xb32-accum16-coslr-200e` : `accum16` 表示权重会在梯度累积16个迭代之后更新。
- `8xb512-amp-coslr-300e` : `amp` 表示使用混合精度训练。

### 数据信息

数据信息包含数据集，输入大小等。例如：

- `in1k` : `ImageNet1k` 数据集，默认使用的输入图像大小是 224x224
- `in1k-384px` : 表示输入图像大小是384x384
- `cifar10`
- `inat18` : `iNaturalist2018` 数据集，包含 8142 类
- `places205`

### 配置文件命名示例

这一节，我们通过一个具体的例子来说明文件命名的规则:

```
swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py
```

- `swav`: 算法信息
- `resnet50`: 模块信息
- `8xb32-mcrop-2-6-coslr-200e`: 训练信息
  - `8xb32`: 共使用 8 张 GPU，每张 GPU 上的 batch size 是 32
  - `mcrop-2-6`: 使用 multi-crop 数据增强方法
  - `coslr`: 使用余弦学习率调度器
  - `200e`: 训练模型200个周期
- `in1k-224-96`: 数据信息，在 ImageNet1k 数据集上训练，输入大小是 224x224 和 96x96

## 配置文件结构

在 `configs/_base_` 文件夹中, 有 4 种类型的基础组件文件，即：

- models
- datasets
- schedules
- runtime

所有的基础配置文件定义了训练所需的最基础的元素，例如 train/val/test 循环，优化器。你可以通过继承一些基础配置文件快捷地构建你自己的配置。由 `_base_` 下的组件组成的配置被称为 原始配置（primitive）。为了易于理解，我们使用 MoCo v2 作为一个例子，并对它的每一行做出注释。若想了解更多细节，请参考 API 文档。

配置文件 `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py` 如下所述：

```python
_base_ = [
    '../_base_/models/mocov2.py',                  # 模型
    '../_base_/datasets/imagenet_mocov2.py',       # 数据
    '../_base_/schedules/sgd_coslr-200e_in1k.py',  # 训练调度
    '../_base_/default_runtime.py',                # 运行时设置
]

# 我们继承了默认的运行时设置，同时修改了  ``CheckpointHook``.
# max_keep_ckpts 控制在 work_dirs 中最多保存多少个 checkpoint 文件
# 例如是 3, ``CheckpointHook`` 将会只保存最近的 3 个 checkpoint 文件
# 如果在 work_dirs 中超过了 3 个文件, 将会自动删掉时间最久远的那个 checkpoint
# , 从而保持 checkpoint 文件的数目始终为 3
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
```

`../_base_/models/mocov2.py` 是 MoCo v2 的基础模型配置。

```python
# type='MoCo' 指代我们使用 MoCo 这个算法。 我们将改算法分为四个部分：
# backbone, neck, head 和 loss。'queue_len', 'feat_dim' and 'momentum' 是另外
# 几个 MoCo 需要的参数。
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.2))
```

`../_base_/datasets/imagenet_mocov2.py` 是 MoCo v2 的基础数据集配置。主要写出了
与 dataset 和 dataloader 相关的信息。

```python
# dataset 配置
# 我们使用 MMClassification 中实现的 ``ImageNet`` dataset 数据集, 所以
# 这里有一个 ``mmcls`` 前缀.
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

# mocov2 和 mocov1 的主要差异在于数据增强的不同
view_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.2, 1.), backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
```

`../_base_/schedules/sgd_coslr-200e_in1k.py` 是 MoCo v2 的基础调度配置。

```python
# 优化器
optimizer = dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# 学习率调整策略
# 使用 cosine learning rate decay
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=200, by_epoch=True, begin=0, end=200)
]

# 循环设置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
```

`../_base_/default_runtime.py` 是运行时的默认配置。 运行时设置主要包含一些训练中需要使用的基础配置, 例如 default_hooks 和 log_processor

```python
default_scope = 'mmselfsup'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    interval=50,
    custom_keys=[dict(data_src='', method='mean', windows_size='global')])

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_level = 'INFO'
load_from = None
resume = False
```

## 继承和修改配置文件

为了易于理解，我们推荐贡献者从现有方法继承。

对于同一个文件夹下的所有配置，我们推荐只使用**一个**原始（primitive） 配置。其他所有配置应当从 原始（primitive） 配置继承。这样最大的继承层次为 3。

例如，如果你的配置文件是基于 MoCo v2 做一些修改， 首先你可以通过指定 `_base_ ='./mocov2_resnet50_8xb32-coslr-200e_in1k.py.py'` （相对于你的配置文件的路径）继承基本的 MoCo v2 结构，接着在配置文件中修改一些必要的参数。 现在，我们举一个更具体的例子，我们想使用 `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py.py`中几乎所有的配置，但是将训练周期数从 200 修改为 800，修改学习率衰减的时机和数据集路径，你可以创建一个名为 `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-800e_in1k.py.py` 的新配置文件，内容如下：

```python
_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'


# 学习率调整器
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=800, by_epoch=True, begin=0, end=800)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800)
```

### 使用配置中的中间变量

在配置文件中使用一些中间变量会使配置文件更加清晰和易于修改。

例如 `dataset_type`, `train_pipeline`, `file_client_args` 是数据中的中间变量。 我们先定义它们再将它们传进 `data`.

```python
# 数据集配置
# 我们使用来源于 MMClassification 中的 ``ImageNet``, 所以有一个 ``mmcls`` 的前缀
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

# mocov2 和 mocov1 的不同主要来自于数据增强
view_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
```

### 忽略基础配置中的字段

有时候，你需要设置 `_delete_=True` 来忽略基础配置文件中一些域的内容。 您可以参考 [mmengine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) 获得更多说明。 接下来是一个例子。如果你希望在 SimCLR 使用中 `MoCoV2Neck`, 仅仅继承并直接修改将会报 `get unexcepected keyword 'num_layers'` 错误， 因为在 `model.neck` 域信息中，基础配置 `num_layers` 字段被保存下来了， 你需要添加 `_delete_=True` 来忽略 `model.neck` 在基础配置文件中的有关字段的内容：

```python
_base_ = 'simclr_resnet50_8xb32-coslr-200e_in1k.py'

model = dict(
    neck=dict(
        _delete_=True,
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True))
```

### 使用基础配置中的字段

有时候，你可能引用 `_base_` 配置中一些字段, 以避免重复定义。 你可以参考 [mmengine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) 获取更多的说明。
下面是一个使用基础配置文件中 `num_classes` 的例子, 请参考 `configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py`.

```python
_base_ = [
    '../_base_/models/odc.py',
    '../_base_/datasets/imagenet_odc.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(num_classes={{_base_.num_classes}}),
    memory_bank=dict(num_classes={{_base_.num_classes}}),
)

```

## 通过脚本参数修改配置

当用户使用脚本 "tools/train.py" 或 "tools/test.py" 提交任务，或者其他工具时，可以通过指定 `--cfg-options` 参数来直接修改配置文件中内容。

- 更新字典链中的配置的键

  配置项可以通过遵循原始配置中键的层次顺序指定。例如，`--cfg-options model.backbone.norm_eval=False` 改变模型 backbones 中的所有 BN 模块为 `train` 模式。

- 更新列表中配置的键

  你的配置中的一些配置字典是由列表组成。例如，训练 pipeline `data.train.pipeline` 通常是一个列表。
  例如 `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`。 如果你想要在 pipeline 中将 `'flip_prob=0.5'` 修改为 `'flip_prob=0.0'` ，
  您可以指定 `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- 更新 list/tuples 中的值

  如果想要更新的值是一个列表或者元组。 例如, 一些配置文件中包含 `param_scheduler = "[dict(type='CosineAnnealingLR',T_max=200,by_epoch=True,begin=0,end=200)]"`。 如果你想要改变这个键，你可以指定 `--cfg-options param_scheduler = "[dict(type='LinearLR',start_factor=1e-4, by_epoch=True,begin=0,end=40,convert_to_iter_based=True)]"`。 注意, " 是必要的, 并且在指定值的时候，在引号中不能存在空白字符。

## 导入用户定义模块

```{note}
这部分内容初学者可以跳过，只在使用其他 MM-codebase 时会用到，例如使用 mmcls 作为第三方库来构建你的工程。
```

这部分内容初学者可以跳过，只在使用其他 MM-codebase 时会用到，例如使用 mmcls 作为第三方库来构建你的工程。 i为了简化代码，你可以使用 MM-codebase 作为第三方库，只需要保存你自己额外的代码，并在配置文件中导入自定义模块。你可以参考 [OpenMMLab Algorithm Competition Project](https://github.com/zhangrui-wolf/openmmlab-competition-2021) 中的例子.

在你自己的配置文件中添加如下所述的代码：

```python
custom_imports = dict(
    imports=['your_dataset_class',
             'your_transforme_class',
             'your_model_class',
             'your_module_class'],
    allow_failed_imports=False)
```
