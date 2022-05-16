# 教程 0: 学习配置

MMSelfSup 主要使用python文件作为配置。我们设计的配置文件系统集成了模块化和继承性，方便用户实施各种实验。所有的配置文件都放在 `configs` 文件夹。如果你想概要地审视配置文件，你可以执行 `python tools/misc/print_config.py` 查看完整配置。

<!-- TOC -->

- [教程 0: 学习配置](#%E6%95%99%E7%A8%8B-0-%E5%AD%A6%E4%B9%A0%E9%85%8D%E7%BD%AE)
  - [配置文件与检查点命名约定](#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E4%B8%8E%E6%A3%80%E6%9F%A5%E7%82%B9%E5%91%BD%E5%90%8D%E7%BA%A6%E5%AE%9A)
    - [算法信息](#%E7%AE%97%E6%B3%95%E4%BF%A1%E6%81%AF)
    - [模块信息](#%E6%A8%A1%E5%9D%97%E4%BF%A1%E6%81%AF)
    - [训练信息](#%E8%AE%AD%E7%BB%83%E4%BF%A1%E6%81%AF)
    - [数据信息](#%E6%95%B0%E6%8D%AE%E4%BF%A1%E6%81%AF)
    - [配置文件命名示例](#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E5%91%BD%E5%90%8D%E7%A4%BA%E4%BE%8B)
    - [检查点命名约定](#%E6%A3%80%E6%9F%A5%E7%82%B9%E5%91%BD%E5%90%8D%E7%BA%A6%E5%AE%9A)
  - [配置文件结构](#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84)
  - [继承和修改配置文件](#%E7%BB%A7%E6%89%BF%E5%92%8C%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)
    - [使用配置中的中间变量](#%E4%BD%BF%E7%94%A8%E9%85%8D%E7%BD%AE%E4%B8%AD%E7%9A%84%E4%B8%AD%E9%97%B4%E5%8F%98%E9%87%8F)
    - [忽略基础配置中的字段](#%E5%BF%BD%E7%95%A5%E5%9F%BA%E7%A1%80%E9%85%8D%E7%BD%AE%E4%B8%AD%E7%9A%84%E5%AD%97%E6%AE%B5)
    - [使用基础配置中的字段](#%E4%BD%BF%E7%94%A8%E5%9F%BA%E7%A1%80%E9%85%8D%E7%BD%AE%E4%B8%AD%E7%9A%84%E5%AD%97%E6%AE%B5)
  - [通过脚本参数修改配置](#%E9%80%9A%E8%BF%87%E8%84%9A%E6%9C%AC%E5%8F%82%E6%95%B0%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE)
  - [导入用户定义模块](#%E5%AF%BC%E5%85%A5%E7%94%A8%E6%88%B7%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9D%97)

<!-- TOC -->

## 配置文件与检查点命名约定

我们遵循下述约定来命名配置文件并建议贡献者也遵循该命名风格。配置文件名字被分成4部分：算法信息、模块信息、训练信息和数据信息。逻辑上，不同部分用下划线连接 `'_'`，同一部分中的单词使用破折线 `'-'` 连接。

```
{algorithm}_{module}_{training_info}_{data_info}.py
```

- `algorithm info`：包含算法名字的算法信息，例如simclr，mocov2等；
- `module info`： 模块信息，用来表示一些 backbone，neck 和 head 信息；
- `training info`：训练信息，即一些训练调度，包括批大小，学习率调度，数据增强等；
- `data info`：数据信息：数据集名字，输入大小等，例如 imagenet，cifar 等。

### 算法信息

```
{algorithm}-{misc}
```

`Algorithm` 表示论文中的算法缩写和版本。例如：

- `relative-loc`：不同单词之间使用破折线连接 `'-'`
- `simclr`
- `mocov2`

`misc` 提供一些其他算法相关信息。例如：

- `npid-ensure-neg`
- `deepcluster-sobel`

### 模块信息

```
{backbone setting}-{neck setting}-{head_setting}
```

模块信息主要包含 backboe 信息。例如：

- `resnet50`
- `vit`（将会用在mocov3中）

或者其他一些需要在配置名字中强调的特殊的设置。例如：

- `resnet50-nofrz`：在一些下游任务的训练中，该 backbone 不会冻结 stages

### 训练信息

训练相关的配置，包括 batch size, lr schedule, data augment 等。

- Batch size，格式是 `{gpu x batch_per_gpu}` ，例如 `8xb32`；
- Training recipe，该方法以如下顺序组织：`{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`

例如：

- `8xb32-mcrop-2-6-coslr-200e`：`mcrop` 是 SwAV 提出的 pipeline 中的名为 multi-crop 的一部分。2 和 6 表示 2 个 pipeline 分别输出 2 个和 6 个裁剪图，而且裁剪信息记录在数据信息中；
- `8xb32-accum16-coslr-200e`：`accum16` 表示权重会在梯度累积16个迭代之后更新。

### 数据信息

数据信息包含数据集，输入大小等。例如：

- `in1k`：`ImageNet1k` 数据集，默认使用的输入图像大小是 224x224
- `in1k-384px`：表示输入图像大小是384x384
- `cifar10`
- `inat18`：`iNaturalist2018` 数据集，包含 8142 类
- `places205`

### 配置文件命名示例

```
swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py
```

- `swav`：算法信息
- `resnet50`：模块信息
- `8xb32-mcrop-2-6-coslr-200e`：训练信息
  - `8xb32`：共使用 8 张 GPU，每张 GPU 上的 batch size 是 32
  - `mcrop-2-6`：使用 multi-crop 数据增强方法
  - `coslr`：使用余弦学习率调度器
  - `200e`：训练模型200个周期
- `in1k-224-96`：数据信息，在 ImageNet1k 数据集上训练，输入大小是 224x224 和 96x96

### 检查点命名约定

权重的命名主要包括配置文件名字，日期和哈希值。

```
{config_name}_{date}-{hash}.pth
```

## 配置文件结构

在 `configs/_base_` 文件中，有 4 种类型的基础组件文件，即

- models
- datasets
- schedules
- runtime

你可以通过继承一些基础配置文件快捷地构建你自己的配置。由 `_base_` 下的组件组成的配置被称为 _原始配置（primitive）_。

为了易于理解，我们使用 MoCo v2 作为一个例子，并对它的每一行做出注释。若想了解更多细节，请参考 API 文档。

配置文件 `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py` 如下所述。

```python
_base_ = [
    '../_base_/models/mocov2.py',                  # 模型
    '../_base_/datasets/imagenet_mocov2.py',       # 数据
    '../_base_/schedules/sgd_coslr-200e_in1k.py',  # 训练调度
    '../_base_/default_runtime.py',                # 运行时设置
]

# 在这里，我们继承运行时设置并修改 max_keep_ckpts。
# max_keep_ckpts 控制在你的 work_dirs 中最大的ckpt文件的数量
# 如果它是3，当 CheckpointHook (在mmcv中) 保存第 4 个 ckpt 时，
# 它会移除最早的那个，使总的 ckpt 文件个数保持为 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
```

```{note}
配置文件中的 'type' 是一个类名，而不是参数的一部分。
```

`../_base_/models/mocov2.py` 是 MoCo v2 的基础模型配置。

```python
model = dict(
    type='MoCo',  # 算法名字
    queue_len=65536,  # 队列中维护的负样本数量
    feat_dim=128,  # 紧凑特征向量的维度，等于 neck 的 out_channels
    momentum=0.999,  # 动量更新编码器的动量系数
    backbone=dict(
        type='ResNet',  # Backbone name
        depth=50,  # backbone 深度，ResNet 可以选择 18、34、50、101、 152
        in_channels=3,  # 输入图像的通道数
        out_indices=[4],  # 输出特征图的输出索引，0 表示 conv-1，x 表示 stage-x
        norm_cfg=dict(type='BN')),  # 构建一个字典并配置 norm 层
    neck=dict(
        type='MoCoV2Neck',  # Neck name
        in_channels=2048,  # 输入通道数
        hid_channels=2048,  # 隐层通道数
        out_channels=128,  # 输出通道数
        with_avg_pool=True),  # 是否在 backbone 之后使用全局平均池化
    head=dict(
        type='ContrastiveHead',  # Head name, 表示 MoCo v2 使用 contrastive loss
        temperature=0.2))  # 控制分布聚集程度的温度超参数
```

`../_base_/datasets/imagenet_mocov2.py` 是 MoCo v2 的基础数据集配置。

```python
# 数据集配置
data_source = 'ImageNet'  # 数据源名字
dataset_type = 'MultiViewDataset' # 组成 pipeline 的数据集类型
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],  # 用来预训练预训练 backboe 模型的均值
    std=[0.229, 0.224, 0.225])  # 用来预训练预训练 backbone 模型的标准差
# mocov2 和 mocov1 之间的差异在于 pipeline 中的 transforms
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),  # RandomResizedCrop
    dict(
        type='RandomAppliedTrans',  # 以0.8的概率随机使用 ColorJitter 增强方法
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),  # 0.2概率的 RandomGrayscale
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),  # 0.5概率的随机 GaussianBlur
    dict(type='RandomHorizontalFlip'),  # 随机水平翻转图像
]

# prefetch
prefetch = False  # 是否使用 prefetch 加速 pipeline
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# 数据集汇总
data = dict(
    samples_per_gpu=32,  # 单张 GPU 的批大小, 共 32*8=256
    workers_per_gpu=4,  # 每张 GPU 用来 pre-fetch 数据的 worker 个数
    drop_last=True,  # 是否丢弃最后一个 batch 的数据
    train=dict(
        type=dataset_type,  # 数据集名字
        data_source=dict(
            type=data_source,  # 数据源名字
            data_prefix='data/imagenet/train',  # 数据集根目录, 当 ann_file 不存在时，类别信息自动从该根目录自动获取
            ann_file='data/imagenet/meta/train.txt',  #  若 ann_file 存在，类别信息从该文件获取
        ),
        num_views=[2],  # pipeline 中不同的视图个数
        pipelines=[train_pipeline],  # 训练 pipeline
        prefetch=prefetch,  # 布尔值
    ))
```

`../_base_/schedules/sgd_coslr-200e_in1k.py` 是 MoCo v2 的基础调度配置。

```python
# 优化器
optimizer = dict(
    type='SGD',  # 优化器类型
    lr=0.03,  # 优化器的学习率, 参数的详细使用请参阅 PyTorch 文档
    weight_decay=1e-4,  # 动量参数
    momentum=0.9)  # SGD 的权重衰减
# 用来构建优化器钩子的配置，请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 中的实现细节。
optimizer_config = dict()  # 这个配置可以设置 grad_clip，coalesce，bucket_size_mb 等。

# 学习策略
# 用来注册 LrUpdater 钩子的学习率调度配置
lr_config = dict(
    policy='CosineAnnealing',  # 调度器策略，也支持 Step，Cyclic 等。 LrUpdater 支持的细节请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9。
    min_lr=0.)  # CosineAnnealing 中的最小学习率设置

# 运行时设置
runner = dict(
    type='EpochBasedRunner',  # 使用的 runner 的类型 (例如 IterBasedRunner 或 EpochBasedRunner)
    max_epochs=200) # 运行工作流周期总数的 Runner 的 max_epochs，对于IterBasedRunner 使用 `max_iters`

```

`../_base_/default_runtime.py` 是运行时的默认配置。

```python
# 保存检查点
checkpoint_config = dict(interval=10)  # 保存间隔是10

# yapf:disable
log_config = dict(
    interval=50,  # 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook'),  # 也支持 Tensorboard logger
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

# 运行时设置
dist_params = dict(backend='nccl') # 设置分布式训练的参数，端口也支持设置。
log_level = 'INFO'  # 日志的输出 level。
load_from = None  # 加载 ckpt
resume_from = None  # 从给定的路径恢复检查点，将会从检查点保存时的周期恢复训练。
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] 表示有一个 workflow，该 workflow 名字是 'train' 且执行一次。
persistent_workers = True  # Dataloader 中设置 persistent_workers 的布尔值，详细信息请参考 PyTorch 文档
```

## 继承和修改配置文件

为了易于理解，我们推荐贡献者从现有方法继承。

对于同一个文件夹下的所有配置，我们推荐只使用**一个** _原始（primitive）_ 配置。其他所有配置应当从  _原始（primitive）_ 配置继承，这样最大的继承层次为 3。

例如，如果你的配置文件是基于 MoCo v2 做一些修改，首先你可以通过指定 `_base_ ='./mocov2_resnet50_8xb32-coslr-200e_in1k.py.py'` （相对于你的配置文件的路径）继承基本的 MoCo v2 结构，数据集和其他训练设置，接着在配置文件中修改一些必要的参数。现在，我们举一个更具体的例子，我们想使用 `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py.py` 中几乎所有的配置，但是将训练周期数从 200 修改为 800，修改学习率衰减的时机和数据集路径，你可以创建一个名为 `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-800e_in1k.py.py` 的新配置文件，内容如下：

```python
_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'

runner = dict(max_epochs=800)
```

### 使用配置中的中间变量

在配置文件中使用一些中间变量会使配置文件更加清晰和易于修改。

例如：数据中的中间变量有 `data_source`, `dataset_type`, `train_pipeline`, `prefetch`. 我们先定义它们再将它们传进 `data`。

```python
data_source = 'ImageNet'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [...]

# prefetch
prefetch = False  # 是否使用 prefetch 加速 pipeline
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(type=dataset_type, type=data_source, data_prefix=...),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))
```

### 忽略基础配置中的字段

有时候，你需要设置 `_delete_=True` 来忽略基础配置文件中一些域的内容。 你可以参考 [mmcv](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) 获得更多说明。

接下来是一个例子。如果你希望在 simclr 的设置中使用 `MoCoV2Neck`，仅仅继承并直接修改将会报 `get unexcepected keyword 'num_layers'` 错误，因为在 `model.neck` 域信息中，基础配置 `'num_layers'` 字段被保存下来了， 你需要添加 `_delete_=True` 来忽略 `model.neck` 在基础配置文件中的有关字段的内容。

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

有时候，你可能引用 `_base_` 配置中一些字段，以避免重复定义。你可以参考[mmcv](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/config.html#reference-variables-from-base) 获取更多的说明。

下面是在训练数据预处理 pipeline 中使用 auto augment 的一个例子，请参考 `configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py`。当定义 `num_classes` 时，只需要将 auto augment 的定义文件名添入到 `_base_`，并使用 `{{_base_.num_classes}}` 来引用这些变量：

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

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.06,
    momentum=0.9,
    weight_decay=1e-5,
    paramwise_options={'\\Ahead.': dict(momentum=0.)})

# learning policy
lr_config = dict(policy='step', step=[400], gamma=0.4)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=440)
# max_keep_ckpts 控制在你的 work_dirs 中保存的 ckpt 的最大数目
# 如果它等于3，CheckpointHook（在mmcv中）在保存第 4 个 ckpt 时，
# 它会移除最早的那个，使总的 ckpt 文件个数保持为 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
```

## 通过脚本参数修改配置

当用户使用脚本 "tools/train.py" 或 "tools/test.py" 提交任务，或者其他工具时，可以通过指定 `--cfg-options` 参数来直接修改配置文件中内容。

- 更新字典链中的配置的键

  配置项可以通过遵循原始配置中键的层次顺序指定。例如，`--cfg-options model.backbone.norm_eval=False` 改变模型 backbones 中的所有 BN 模块为 `train` 模式。

- 更新列表中配置的键

  你的配置中的一些配置字典是由列表组成。例如，训练 pipeline `data.train.pipeline` 通常是一个列表。例如 `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`。如果你想要在 pipeline 中将 `'flip_prob=0.5'` 修改为 `'flip_prob=0.0'`，你可以指定 `--cfg-options data.train.pipeline.1.flip_prob=0.0`

- 更新 list/tuples 中的值

  如果想要更新的值是一个列表或者元组，例如：配置文件通常设置 `workflow=[('train', 1)]`。如果你想要改变这个键，你可以指定 `--cfg-options workflow="[(train,1),(val,1)]"`。注意：对于 list/tuple 数据类型，引号" 是必须的，并且在指定值的时候，在引号中 **NO** 空白字符。

## 导入用户定义模块

```{note}
这部分内容初学者可以跳过，只在使用其他 MM-codebase 时会用到，例如使用 mmcls 作为第三方库来构建你的工程。
```

你可能使用其他的 MM-codebase 来完成你的工程，并在工程中创建新的数据集类，模型类，数据增强类等。为了简化代码，你可以使用 MM-codebase 作为第三方库，只需要保存你自己额外的代码，并在配置文件中导入自定义模块。你可以参考 [OpenMMLab Algorithm Competition Project](https://github.com/zhangrui-wolf/openmmlab-competition-2021) 中的例子。

在你自己的配置文件中添加如下所述的代码：

```python
custom_imports = dict(
    imports=['your_dataset_class',
             'your_transforme_class',
             'your_model_class',
             'your_module_class'],
    allow_failed_imports=False)
```
