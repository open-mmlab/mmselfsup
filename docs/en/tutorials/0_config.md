# Tutorial 0: Learn about Configs

MMSelfSup mainly uses python files as configs. The design of our configuration file system integrates modularity and inheritance, facilitating users to conduct various experiments. All configuration files are placed in the `configs` folder. If you wish to inspect the config file in summary, you may run `python tools/misc/print_config.py` to see the complete config.

<!-- TOC -->

- [Tutorial 0: Learn about Configs](#tutorial-0-learn-about-configs)
  - [Config File and Checkpoint Naming Convention](#config-file-and-checkpoint-naming-convention)
    - [Algorithm information](#algorithm-information)
    - [Module information](#module-information)
    - [Training information](#training-information)
    - [Data information](#data-information)
    - [Config File Name Example](#config-file-name-example)
    - [Checkpoint Naming Convention](#checkpoint-naming-convention)
  - [Config File Structure](#config-file-structure)
  - [Inherit and Modify Config File](#inherit-and-modify-config-file)
    - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)
    - [Ignore some fields in the base configs](#ignore-some-fields-in-the-base-configs)
    - [Use some fields in the base configs](#use-some-fields-in-the-base-configs)
  - [Modify config through script arguments](#modify-config-through-script-arguments)
  - [Import user-defined modules](#import-user-defined-modules)

<!-- TOC -->

## Config File and Checkpoint Naming Convention

We follow the below convention to name config files. Contributors are advised to follow the same style. The config file names are divided into four parts: algorithm info, module information, training information and data information. Logically, different parts are concatenated by underscores `'_'`, and words in the same part are concatenated by dashes `'-'`.

```
{algorithm}_{module}_{training_info}_{data_info}.py
```

- `algorithm info`：Algorithm information includes algorithm name, such as simclr, mocov2, etc.;
- `module info`： Module information is used to represent some backbone, neck, head information;
- `training info`：Training information, some training schedule, including batch size, lr schedule, data augment and the like;
- `data info`：Data information, dataset name, input size and so on, such as imagenet, cifar, etc.;

### Algorithm information

```
{algorithm}-{misc}
```

`Algorithm` means the abbreviation from the paper and its version. E.g:

- `relative-loc` : The different word is concatenated by dashes `'-'`
- `simclr`
- `mocov2`

`misc` offers some other algorithm related information. E.g.

- `npid-ensure-neg`
- `deepcluster-sobel`

### Module information

```
{backbone setting}-{neck setting}-{head_setting}
```

The module information mainly includes the backbone information. E.g:

- `resnet50`
- `vit`（will be used in mocov3）

Or there are some special settings which is needed to be mentioned in the config name. E.g:

- `resnet50-nofrz`: In some downstream tasks，the backbone will not froze stages while training

### Training information

Training related settings，including batch size, lr schedule, data augment, etc.

- Batch size, the format is `{gpu x batch_per_gpu}`，like `8xb32`;
- Training recipe，the methods will be arranged in the order `{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`.

E.g:

- `8xb32-mcrop-2-6-coslr-200e` : `mcrop` is proposed in SwAV named multi-crop，part of pipeline. 2 and 6 means that 2 pipelines will output 2 and 6 crops correspondingly，the crop size is recorded in data information;
- `8xb32-accum16-coslr-200e` : `accum16` means the gradient will accumulate for 16 iterations，then the weights will be updated.

### Data information

Data information contains the dataset, input size, etc. E.g:

- `in1k` : `ImageNet1k` dataset, default to use the input image size of 224x224
- `in1k-384px` : Indicates that the input image size is 384x384
- `cifar10`
- `inat18` : `iNaturalist2018` dataset，it has 8142 classes
- `places205`

### Config File Name Example

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
- `in1k-224-96`: Data information，train on ImageNet1k dataset，the input sizes are 224x224 and 96x96

### Checkpoint Naming Convention

The naming of the weight mainly includes the configuration file name, date and hash value.

```
{config_name}_{date}-{hash}.pth
```

## Config File Structure

There are four kinds of basic component file in the `configs/_base_` folders, namely：

- models
- datasets
- schedules
- runtime

You can easily build your own training config file by inherit some base config files. And the configs that are composed by components from `_base_` are called _primitive_.

For easy understanding, we use MoCo v2 as a example and comment the meaning of each line. For more detaile, please refer to the API documentation.

The config file `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py` is displayed below.

```python
_base_ = [
    '../_base_/models/mocov2.py',                  # model
    '../_base_/datasets/imagenet_mocov2.py',       # data
    '../_base_/schedules/sgd_coslr-200e_in1k.py',  # training schedule
    '../_base_/default_runtime.py',                # runtime setting
]

# Here we inherit runtime settings and modify the max_keep_ckpts.
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
```

```{note}
The 'type' in the configuration file is not a constructed parameter, but a class name.
```

`../_base_/models/mocov2.py` is the base model config for MoCo v2.

```python
model = dict(
    type='MoCo',  # Algorithm name
    queue_len=65536,  # Number of negative keys maintained in the queue
    feat_dim=128,  # Dimension of compact feature vectors, equal to the out_channels of the neck
    momentum=0.999,  # Momentum coefficient for the momentum-updated encoder
    backbone=dict(
        type='ResNet',  # Backbone name
        depth=50,  # Depth of backbone, ResNet has options of 18, 34, 50, 101, 152
        in_channels=3,  # The channel number of the input images
        out_indices=[4],  # The output index of the output feature maps, 0 for conv-1, x for stage-x
        norm_cfg=dict(type='BN')),  # Dictionary to construct and config norm layer
    neck=dict(
        type='MoCoV2Neck',  # Neck name
        in_channels=2048,  # Number of input channels
        hid_channels=2048,  # Number of hidden channels
        out_channels=128,  # Number of output channels
        with_avg_pool=True),  # Whether to apply the global average pooling after backbone
    head=dict(
        type='ContrastiveHead',  # Head name, indicates that the MoCo v2 use contrastive loss
        temperature=0.2))  # The temperature hyper-parameter that controls the concentration level of the distribution.
```

`../_base_/datasets/imagenet_mocov2.py` is the base dataset config for MoCo v2.

```python
# dataset settings
data_source = 'ImageNet'  # data source name
dataset_type = 'MultiViewDataset' # dataset type is related to the pipeline composing
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],  # Mean values used to pre-training the pre-trained backbone models
    std=[0.229, 0.224, 0.225])  # Standard variance used to pre-training the pre-trained backbone models
# The difference between mocov2 and mocov1 is the transforms in the pipeline
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),  # RandomResizedCrop
    dict(
        type='RandomAppliedTrans',  # Random apply ColorJitter augment method with probability 0.8
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),  # RandomGrayscale with probability 0.2
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),  # Random GaussianBlur with probability 0.5
    dict(type='RandomHorizontalFlip'),  # Randomly flip the picture horizontally
]

# prefetch
prefetch = False  # Whether to using prefetch to speed up the pipeline
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,  # Batch size of a single GPU, total 32*8=256
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    drop_last=True,  # Whether to drop the last batch of data
    train=dict(
        type=dataset_type,  # dataset name
        data_source=dict(
            type=data_source,  # data source name
            data_prefix='data/imagenet/train',  # Dataset root, when ann_file does not exist, the category information is automatically obtained from the root folder
            ann_file='data/imagenet/meta/train.txt',  #  ann_file existes, the category information is obtained from file
        ),
        num_views=[2],  # The number of different views from pipeline
        pipelines=[train_pipeline],  # The train pipeline
        prefetch=prefetch,  # The boolean value
    ))
```

`../_base_/schedules/sgd_coslr-200e_in1k.py` is the base schedule config for MoCo v2.

```python
# optimizer
optimizer = dict(
    type='SGD',  # Optimizer type
    lr=0.03,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
    weight_decay=1e-4,  # Momentum parameter
    momentum=0.9)  # Weight decay of SGD
# Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
optimizer_config = dict()  # this config can set grad_clip, coalesce, bucket_size_mb, etc.

# learning policy
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(
    policy='CosineAnnealing',  # The policy of scheduler, also support Step, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    min_lr=0.)  # The minimum lr setting in CosineAnnealing

# runtime settings
runner = dict(
    type='EpochBasedRunner',  # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=200) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`

```

`../_base_/default_runtime.py` is the default runtime settings.

```python
# checkpoint saving
checkpoint_config = dict(interval=10)  # The save interval is 10

# yapf:disable
log_config = dict(
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),  # The Tensorboard logger is also supported
        # dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl') # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The output level of the log.
load_from = None  # Runner to load ckpt
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
persistent_workers = True  # The boolean type to set persistent_workers in Dataloader. see detail in the documentation of PyTorch
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors to inherit from existing methods.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For example, if your config file is based on MoCo v2 with some other modification, you can first inherit the basic MoCo v2 structure, dataset and other training setting by specifying `_base_ ='./mocov2_resnet50_8xb32-coslr-200e_in1k.py.py'` (The path relative to your config file), and then modify the necessary parameters in the config file. A more specific example, now we want to use almost all configs in `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py.py`, but change the number of training epochs from 200 to 800, modify when to decay the learning rate, and modify the dataset path, you can create a new config file `configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-800e_in1k.py.py` with content as below:

```python
_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'

runner = dict(max_epochs=800)
```

### Use intermediate variables in configs

Some intermediate variables are used in the configuration file. The intermediate variables make the configuration file clearer and easier to modify.

For example, `data_source`, `dataset_type`, `train_pipeline`, `prefetch` are the intermediate variables of the data. We first need to define them and then pass them to `data`.

```python
data_source = 'ImageNet'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [...]

# prefetch
prefetch = False  # Whether to using prefetch to speed up the pipeline
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

### Ignore some fields in the base configs

Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file. You can refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) for more instructions.

The following is an example. If you want to use `MoCoV2Neck` in simclr setting, just using inheritance and directly modify it will report `get unexcepected keyword 'num_layers'` error, because the `'num_layers'` field of the basic config in `model.neck` domain information is reserved, and you need to add `_delete_=True` to ignore the content of `model.neck` related fields in the basic configuration file:

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

Sometimes, you may refer to some fields in the `_base_` config, so as to avoid duplication of definitions. You can refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#reference-variables-from-base) for some more instructions.

The following is an example of using auto augment in the training data preprocessing pipeline， refer to `configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py`. When defining `num_classes`, just add the definition file name of auto augment to `_base_`, and then use `{{_base_.num_classes}}` to reference the variables:

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
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
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

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark " is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Import user-defined modules

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
