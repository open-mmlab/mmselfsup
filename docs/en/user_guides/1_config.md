# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments. If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config. You may also pass `--cfg-options xxx.yyy=zzz` to see the updated config.

- [Tutorial 1: Learn about Configs](#tutorial-1-learn-about-configs)
  - [Config File and Checkpoint Naming Convention](#config-file-and-checkpoint-naming-convention)
    - [Algorithm information](#algorithm-information)
    - [Module information](#module-information)
    - [Training information](#training-information)
    - [Data information](#data-information)
    - [Config file name example](#config-file-name-example)
  - [Config File Structure](#config-file-structure)
  - [Inherit and Modify Config File](#inherit-and-modify-config-file)
    - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)
    - [Ignore some fields in the base configs](#ignore-some-fields-in-the-base-configs)
    - [Reuse some fields in the base configs](#reuse-some-fields-in-the-base-configs)
  - [Modify Config through Script Arguments](#modify-config-through-script-arguments)
  - [Import Modules from Other MM-Codebases](#import-modules-from-other-mm-codebases)

## Config File and Checkpoint Naming Convention

We follow the convention below to name config files. Contributors are advised to follow the same convention. The name of config file is divided into four parts: `algorithm info`, `module information`, `training information` and `data information`. Logically, different parts are connected with underscore `_`, and info belonging to the same part is connected with dash `-`.

The following example is for illustration:

```
{algorithm_info}_{module_info}_{training_info}_{data_info}.py
```

- `algorithm_info`：Algorithm information includes the algorithm name, such as `simclr`, `mocov2`, etc.
- `module_info`： Module information denotes backbones, necks, heads and losses.
- `training_info`：Training information denotes some training schedules, such as batch size, lr schedule, data augmentation, etc.
- `data_info`：Data information includes the dataset name, input size, etc.

We detail the naming convention for each part in the name of the config file:

### Algorithm information

```
{algorithm}-{misc}
```

`algorithm` generally denotes the abbreviation for the paper and its version. E.g.:

- `relative-loc`
- `simclr`
- `mocov2`

`misc` provides some other algorithm-related information. E.g.:

- `npid-ensure-neg`
- `deepcluster-sobel`

Note that different words are connected with dash `-`.

### Module information

```
{backbone_setting}-{neck_setting}-{head_setting}-{loss_setting}
```

The module information mainly includes the backbone information. E.g.:

- `resnet50`
- `vit-base-p16`
- `swin-base`

Sometimes, there are some special settings needed to be mentioned in the config name. E.g.:

- `resnet50-sobel`: In some downstream tasks like linear evaluation, when loading the DeepCluster pre-traiend model, the backbone only takes 2-channel images after the Sobel layer as input.

Note that `neck_setting`, `head_setting` and `loss_setting` are optional.

### Training information

Training related settings，including batch size, lr schedule, data augmentation, etc.

- Batch size: the format is `{gpu x batch_per_gpu}`，e.g., `8xb32`.
- Training recipes: they will be arranged in the order `{pipeline aug}-{train aug}-{scheduler}-{epochs}`.

E.g.:

- `8xb32-mcrop-2-6-coslr-200e` : `mcrop` is the multi-crop data augmentation proposed in SwAV. 2 and 6 means that two pipelines output 2 and 6 crops, respectively. The crop size is recorded in data information.
- `8xb32-accum16-coslr-200e` : `accum16` means the weights will be updated after the gradient is accumulated for 16 iterations.
- `8xb512-amp-coslr-300e` : `amp` denotes the automatic mixed precision training.

### Data information

Data information contains the dataset name, input size, etc. E.g.:

- `in1k` : `ImageNet1k` dataset. The input image size is 224x224 by default
- `in1k-384` : `ImageNet1k` dataset with the input image size of 384x384
- `in1k-384x224` : `ImageNet1k` dataset with the input image size of 384x224 (`HxW`)
- `cifar10`
- `inat18` : `iNaturalist2018` dataset. It has 8142 classes.
- `places205`

### Config File Name Example

Here, we give a specific file name to explain the naming convention.

```
swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py
```

- `swav`: Algorithm information
- `resnet50`: Module information.
- `8xb32-mcrop-2-6-coslr-200e`: Training information
  - `8xb32`: Use 8 GPUs in total，and the batch size is 32 per GPU
  - `mcrop-2-6`:Use the multi-crop data augmentation
  - `coslr`: Use the cosine learning rate decay scheduler
  - `200e`: Train the model for 200 epochs
- `in1k-224-96`: Data information. The model is trained on ImageNet1k dataset with the input size of 224x224 (for 2 crops) and 96x96 (for 6 crops).

## Config File Structure

There are four kinds of basic files in the `configs/_base_`, namely：

- models
- datasets
- schedules
- runtime

All these basic files define the basic elements, such as train/val/test loop and optimizer, to run the experiment.
You can easily build your own training config file by inheriting some base config files. The configs that are composed by components from `_base_` are called _primitive_.

For easy understanding, we use MoCo v2 as an example and comment the meaning of each line. For more details, please refer to the API documentation.

The config file `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py` is displayed below.

```python
_base_ = [
    '../_base_/models/mocov2.py',                  # model
    '../_base_/datasets/imagenet_mocov2.py',       # data
    '../_base_/schedules/sgd_coslr-200e_in1k.py',  # training schedule
    '../_base_/default_runtime.py',                # runtime setting
]

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
```

`../_base_/models/mocov2.py` is the base configuration file for the model of MoCo v2.

```python
# model settings
# type='MoCo' specifies we will use the model of MoCo. And we
# split the model into four parts, which are backbone, neck, head
# and loss. 'queue_len', 'feat_dim' and 'momentum' are required
# by MoCo during the training process.
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

`../_base_/datasets/imagenet_mocov2.py` is the base configuration file for
the dataset of MoCo v2. The configuration file specifies the configuration
for dataset and dataloader.

```python
# dataset settings
# We use the ``ImageNet`` dataset implemented by mmclassification, so there
# is a ``mmcls`` prefix.
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')
# Since we use ``ImageNet`` from mmclassification, we need to set the
# custom_imports here.
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)
# The difference between mocov2 and mocov1 is the transforms in the pipeline
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

`../_base_/schedules/sgd_coslr-200e_in1k.py` is the base configuration file for
the training schedules of MoCo v2.

```python
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
# use cosine learning rate decay here
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=200, by_epoch=True, begin=0, end=200)
]

# loop settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
```

`../_base_/default_runtime.py` contains the default runtime settings. The runtime settings include some basic components during training, such as default_hooks and log_processor

```python
default_scope = 'mmselfsup'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
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
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', windows_size='global')])

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
# custom_hooks = [dict(type='SelfSupVisualizationHook', interval=1)]

log_level = 'INFO'
load_from = None
resume = False
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors to inherit from existing configurations.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For example, if your config file is based on MoCo v2 with some other modifications, you can first inherit the basic configuration of MoCo v2 by specifying `_base_ ='./mocov2_resnet50_8xb32-coslr-200e_in1k.py'` (The path relative to your config file), and then modify the necessary fields in your customized config file. A more specific example, now we want to use almost all configs in `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py`, except for changing the training epochs from 200 to 800, you can create a new config file `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-800e_in1k.py` with the content as below:

```python
_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=800, by_epoch=True, begin=0, end=800)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800)
```

### Use intermediate variables in configs

Some intermediate variables are used in the config file. The intermediate variables make the config file clearer and easier to modify.

For example, `dataset_type`, `data_root`, `train_pipeline` are the intermediate variables of `dataset`. We first need to define them and then pass them into `dataset`.

```python
# dataset settings

# Since we use ``ImageNet`` from mmclassification, we need to set the
# custom_imports here.
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)

# We use the ``ImageNet`` dataset implemented by mmclassification, so there
# is a ``mmcls`` prefix.
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

# The difference between mocov2 and mocov1 is the transforms in the pipeline
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

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of the fields in base configs. You can refer to [mmengine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) for more instructions.

The following is an example. If you want to use `MoCoV2Neck` in SimCLR, directly inheriting and modifying it will report `get unexcepected keyword 'num_layers'` error since `NonLinearNeck` and `MoCoV2Neck` use different keywords to construct. In this case, adding `_delete_=True` would replace all old keys in `neck` field with new keys:

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

### Reuse some fields in the base configs

Sometimes, you may reuse some fields in base configs, so as to avoid duplication of variables. You can refer to [mmengine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) for more instructions.

The following is an example of reusing the `num_classes` variable in the base config file. Please refer to `configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py` for more details.

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

## Modify Config through Script Arguments

When using the script `tools/train.py`/`tools/test.py` to submit tasks or using some other tools, you can directly modify the content of the configuration file by specifying the `--cfg-options` parameter.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes all BN modules in backbone to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`. If you want to change `'flip_prob=0.5'` to `'flip_prob=0.0'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, some config files contain `param_scheduler = "[dict(type='CosineAnnealingLR',T_max=200,by_epoch=True,begin=0,end=200)]"`. If you want to change this key, you may specify `--cfg-options param_scheduler = "[dict(type='LinearLR',start_factor=1e-4, by_epoch=True,begin=0,end=40,convert_to_iter_based=True)]"`. Note that the quotation mark `"` is necessary to support list/tuple data types, and that **NO** white space is allowed inside the quotation marks for the specified value.

```{note}
    This modification only supports modifying configuration items of string, int, float, boolean, None, list and tuple types.
    More specifically, for list and tuple types, the elements inside them must also be one of the above seven types.
```

## Import Modules from Other MM-Codebases

```{note}
This part may only be used when using other MM-codebase, like mmcls as a third party library to build your own project, and beginners can skip it.
```

You may use other MM-codebase to complete your project and create new classes of datasets, models, data enhancements, etc. in the project. In order to streamline the code, you can use MM-codebase as a third-party library, you just need to keep your own extra code and import your own custom module in the config files. For example, you may refer to [OpenMMLab Algorithm Competition Project](https://github.com/zhangrui-wolf/openmmlab-competition-2021) .

Add the following code to your own config files:

```python
custom_imports = dict(
    imports=['your_dataset_class',
             'your_transform_class',
             'your_model_class',
             'your_module_class'],
    allow_failed_imports=False)
```
