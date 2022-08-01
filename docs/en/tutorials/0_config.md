# Tutorial 0: Learn about Configs

MMSelfSup mainly uses python files as configs. The design of our configuration file system integrates modularity and inheritance, facilitating users to conduct various experiments. All configuration files are placed in the `configs` folder. If you wish to inspect the config file in summary, you may run `python tools/misc/print_config.py` to see the complete config.

- [Tutorial 0: Learn about Configs](#tutorial-0-learn-about-configs)
  - [Config File and Checkpoint Naming Convention](#config-file-and-checkpoint-naming-convention)
    - [Algorithm information](#algorithm-information)
    - [Module information](#module-information)
    - [Training information](#training-information)
    - [Data information](#data-information)
    - [Config File Name Example](#config-file-name-example)
  - [Config File Structure](#config-file-structure)
  - [Inherit and Modify Config File](#inherit-and-modify-config-file)
    - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)
    - [Ignore some fields in the base configs](#ignore-some-fields-in-the-base-configs)
    - [Use some fields in the base configs](#use-some-fields-in-the-base-configs)
  - [Modify config through script arguments](#modify-config-through-script-arguments)
  - [Import modules from other MM-codebases](#import-modules-from-other-mm-codebases)

## Config File and Checkpoint Naming Convention

We follow conventions below to name config files. Contributors are advised to follow the same conventions. The name of config file is divided into four parts: `algorithm info`, `module information`, `training information` and `data information`. Logically, different parts are concatenated by underscores `'_'`, and info belonging to the same part is concatenated by dashes `'-'`.

The following example is for illustration:

```
{algorithm_info}_{module_info}_{training_info}_{data_info}.py
```

- `algorithm_info`：Algorithm information includes algorithm name, such as simclr, mocov2, etc;
- `module_info`： Module information denotes backbones, necks, heads and losses;
- `training_info`：Training information, e.g. some training schedules, including batch size, lr schedule, data augment;
- `data_info`：Data information, e.g. dataset name, input size;

We detail the naming convention for each part in the name of the config file:

### Algorithm information

```
{algorithm}-{misc}
```

`algorithm` generally denotes the abbreviation for the paper and its version. For example:

- `relative-loc` : The different word is concatenated by dashes `'-'`
- `simclr`
- `mocov2`

`misc` offers some other algorithm related information.

- `npid-ensure-neg`
- `deepcluster-sobel`

### Module information

```
{backbone_setting}-{neck_setting}-{head_setting}-{loss_setting}
```

The module information mainly includes the backbone information. E.g:

- `resnet50`
- `vit`（will be used in mocov3）

Or there are some special settings which is needed to be mentioned in the config name. E.g:

- `resnet50-nofrz`: In some downstream tasks，the backbone will not froze stages while training

While `neck_setting`, `head_setting` and `loss_setting` are optional.

### Training information

Training related settings，including batch size, lr schedule, data augment, etc.

- Batch size, and the format is `{gpu x batch_per_gpu}`，like `8xb32`;
- Training recipes, and they will be arranged in the order `{pipeline aug}-{train aug}-{scheduler}-{epochs}`.

E.g:

- `8xb32-mcrop-2-6-coslr-200e` : `mcrop` is proposed in SwAV named multi-crop，part of pipeline. 2 and 6 means that 2 pipelines will output 2 and 6 crops correspondingly，the crop size is recorded in data information;
- `8xb32-accum16-coslr-200e` : `accum16` means the gradient will accumulate for 16 iterations，then the weights will be updated.

### Data information

Data information contains the dataset, input size, etc. E.g:

- `in1k` : `ImageNet1k` dataset, default to use the input image size of 224x224
- `in1k-384px` : Indicates that the input image size is 384x384
- `cifar10`
- `inat18` : `iNaturalist2018` dataset, and it has 8142 classes
- `places205`

### Config File Name Example

Here, we give a concret file name to explain the naming conventions.

```
swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py
```

- `swav`: Algorithm information
- `resnet50`: Module information
- `8xb32-mcrop-2-6-coslr-200e`: Training information
  - `8xb32`: Use 8 GPUs in total，and the batch size is 32 per GPU
  - `mcrop-2-6`:Use multi-crop data augment method
  - `coslr`: Use cosine learning rate scheduler
  - `200e`: Train the model for 200 epoch
- `in1k-224-96`: Data information，trained on ImageNet1k dataset，and the input sizes are 224x224 and 96x96

## Config File Structure

There are four kinds of basic files in the `configs/_base_`, namely：

- models
- datasets
- schedules
- runtime

All these basic files define the basic elements, such as train/val/test loop and optimizer, to run the experiment.
You can easily build your own training config file by inheriting some base config files. And the configs that are composed by components from `_base_` are called _primitive_.

For easy understanding, we use MoCo v2 as a example and comment the meaning of each line. For more detaile, please refer to the API documentation.

The config file `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py` is displayed below.

```python
_base_ = [
    '../_base_/models/mocov2.py',                  # model
    '../_base_/datasets/imagenet_mocov2.py',       # data
    '../_base_/schedules/sgd_coslr-200e_in1k.py',  # training schedule
    '../_base_/default_runtime.py',                # runtime setting
]

# Here we inherit the default runtime settings and modify the ``CheckpointHook``.
# The max_keep_ckpts controls the max number of ckpt file in your work_dirs
# If it is 3, the ``CheckpointHook`` will save the latest 3 checkpoints.
# If there are more than 3 checkpoints in work_dirs, it will remove the oldest
# one to keep the total number as 3.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3)
)
```

`../_base_/models/mocov2.py` is the base configuration file for the model of MoCo v2.

```python
# type='MoCo' specifies we will use the model of MoCo. And we
# split the model into four parts, which are backbone, neck, head
# and loss. 'queue_len', 'feat_dim' and 'momentum' are required
# by MoCo during the training process.
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
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
    head=dict(type='ContrastiveHead', temperature=0.2),
    loss=dict(type='mmcls.CrossEntropyLoss'))
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

# Since we use ``ImageNet`` from mmclassification, we need set the
# custom_imports here.
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)
# The difference between mocov2 and mocov1 is the transforms in the pipeline
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
optimizer_wrapper = dict(optimizer=dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9))

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
# custom_hooks = [dict(type='SelfSupVisualizationHook', interval=10)]

log_level = 'INFO'
load_from = None
resume = False
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors to inherit from existing configurations.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For example, if your config file is based on MoCo v2 with some other modification, you can first inherit the basic configuration of MoCo v2 by specifying `_base_ ='./mocov2_resnet50_8xb32-coslr-200e_in1k.py.py'` (The path relative to your config file), and then modify the necessary parameters in your customized config file. A more specific example, now we want to use almost all configs in `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py.py`, except for changing the training epochs from 200 to 800, you can create a new config file `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-800e_in1k.py.py` with content as below:

```python
_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'

runner = dict(max_epochs=800)
```

### Use intermediate variables in configs

Some intermediate variables are used in the configuration file. The intermediate variables make the configuration file more clear and easier to modify.

For example, `dataset_type`, `train_pipeline`, `file_client_args` are the intermediate variables of the data. We first need to define them and then pass them to `data`.

```python
# dataset settings

# Since we use ``ImageNet`` from mmclassification, we need set the
# custom_imports here.
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)

# We use the ``ImageNet`` dataset implemented by mmclassification, so there
# is a ``mmcls`` prefix.
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

# The difference between mocov2 and mocov1 is the transforms in the pipeline
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
```

### Ignore some fields in the base configs

Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file. You can refer to [mmengine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) for more instructions.

The following is an example. If you want to use `MoCoV2Neck` in simclr, just using inheritance and directly modifying it will report `get unexcepected keyword 'num_layers'` error, because the `'num_layers'` field of the basic config in `model.neck` domain information is reserved, and you need to add `_delete_=True` to ignore the original content of `model.neck` in the basic configuration file:

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

### Use some fields in the base configs

Sometimes, you may refer to some fields in the `_base_` config, so as to avoid duplication of definitions. You can refer to [mmengine](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/config.md) for some more instructions.

The following is an example of using the `num_classes` variable in the base configuration file, please refer to `configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py`.

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

## Modify config through script arguments

When users use the script "tools/train.py" or "tools/test.py" to submit tasks or use some other tools, they can directly modify the content of the configuration file used by specifying the `--cfg-options` parameter.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`. If you want to change `'flip_prob=0.5'` to `'flip_prob=0.0'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, some configuration files contain `param_scheduler = "[dict(type='CosineAnnealingLR',T_max=200,by_epoch=True,begin=0,end=200)]"`. If you want to change this key, you may specify `--cfg-options param_scheduler = "[dict(type='LinearLR',start_factor=1e-4, by_epoch=True,begin=0,end=40,convert_to_iter_based=True)]"`. Note that the quotation mark " is necessary to support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Import modules from other MM-codebases

```{note}
This part may only be used when using other MM-codebase, like mmcls as a third party library to build your own project, and beginners can skip it.
```

You may use other MM-codebase to complete your project and create new classes of datasets, models, data enhancements, etc. in the project. In order to streamline the code, you can use MM-codebase as a third-party library, you just need to keep your own extra code and import your own custom module in the configuration files. For examples, you may refer to [OpenMMLab Algorithm Competition Project](https://github.com/zhangrui-wolf/openmmlab-competition-2021) .

Add the following code to your own configuration files:

```python
custom_imports = dict(
    imports=['your_dataset_class',
             'your_transforme_class',
             'your_model_class',
             'your_module_class'],
    allow_failed_imports=False)
```
