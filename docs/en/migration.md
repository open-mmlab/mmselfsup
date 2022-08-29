# Migration

- [Migration](#migration)
  - [Migration from MMSelfSup 0.x](#migration-from-mmselfsup-0x)
    - [Config](#config)
      - [Datasets](#datasets)
      - [Models](#models)
      - [Schedules](#schedules)
    - [Folders and Files](#folders-and-files)
  - [Differences between MMSelfSup and OpenSelfSup](#differences-between-mmselfsup-and-openselfsup)
    - [Modular Design](#modular-design)
      - [Datasets](#datasets-1)
      - [Models](#models-1)
    - [Codebase Conventions](#codebase-conventions)
      - [Configs](#configs)
      - [Scripts](#scripts)

## Migration from MMSelfSup 0.x

we introduce some modifications of MMSelfSup 1.x, to help users to migrate their projects based on MMSelfSup 0.x to 1.x smoothly.

### Config

This section illustrates the changes of our config files in `_base_` folder, which includes three parts

- Datasets: `mmselfsup/configs/selfsup/_base_/datasets`
- Models: `mmselfsup/configs/selfsup/_base_/models`
- Schedules: `mmselfsup/configs/selfsup/_base_/schedules`

#### Datasets

In **MMSelfSup 0.x**, we use key `data` to summarize all information, such as `samples_per_gpu`, `train`, `val`, etc.

For example:

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

In **MMSelfSup 1.x**, we separate `train_dataloader`, `val_dataloader` to summarize information correspodingly and the key `data` has been **removed**.

Here is an example of `train_dataloader`:

```python
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
val_dataloader = ...
```

Besides, we remove the key of `data_source` to keep the consistent pipeline format with other OpenMMLab projects. Please refer to [1_config.md](user_guides/1_config.md) for more details.

#### Models

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

#### Schedules

| MMSelfSup 0.x    | MMSelfSup 1.x   | Remark                                                                                                                          |
| ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| optimizer_config | /               | It has been **removed**.                                                                                                        |
| /                | optim_wrapper   | The `optim_wrapper` provides a common interface for updating parameters.                                                        |
| lr_config        | param_scheduler | The `param_scheduler` is a list to set learning rate or other parameters, which is more flexible.                               |
| runner           | train_cfg       | The loop setting (`EpochBasedTrainLoop`, `IterBasedTrainLoop`) in `train_cfg` controls the work flow of the algorithm training. |

### Folders and Files

The table below records the general modification of the folders and files.

| MMSelfSup 0.x         | MMSelfSup 1.x       | Remark                                                                                                                                                    |
| --------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| apis                  | /                   | Currently, the `apis` folder has been **removed**, it might be added in the future.                                                                       |
| core                  | engine              | The `core` folder has been renamed to `engine`, which includes `hooks`, `opimizers`.                                                                      |
| datasets              | datasets            | The datasets is implemented according to different datasets, such as ImageNet, Places205.                                                                 |
| datasets/data_sources | /                   | The `data_sources` has been **removed** and the directory of `datasets` now is consistent with other OpenMMLab projects.                                  |
| datasets/pipelines    | datasets/transforms | The `pipelines` folder has been renamed to `transforms`.                                                                                                  |
| /                     | evaluation          | The `evaluation` is created for some evaluation functions or classes, such as KNN function or layer for detection.                                        |
| /                     | models/losses       | The `losses` folder is created to provide different loss implementation, which is from `heads`                                                            |
| /                     | structures          | The `structures` folder is for the implementation of data structures. In MMSelfSup, we provide `selfsup_data_sample` to store different data information. |
| /                     | visualization       | The `visualization` folder contains the visualizer, which is responsible for some visualization tasks like visualizing data augmentation.                 |

## Differences between MMSelfSup and OpenSelfSup

This file records differences between the newest version of MMSelfSup with older versions and OpenSelfSup.

MMSelfSup goes through a refactoring and addresses many legacy issues. It is not compatitible with OpenSelfSup, i.e. the old config files are supposed to be updated as some arguments of the class or names of the components have been modified.

The major differences are in two folds: codebase conventions, modular design.

### Modular Design

In order to build more clear directory structure, MMSelfSup redesigns some of the modules.

#### Datasets

- MMSelfSup merges some datasets to reduce some redundant codes.

  - Classification, Extraction, NPID -> OneViewDataset

  - BYOL, Contrastive -> MultiViewDataset

- The `data_sources` folder has been refactored, thus the loading function is more robust.

In addition, this part is still under refactoring, it will be released in following version.

#### Models

- The registry mechanism is updated. Currently, the parts under the `models` folder are built with a parent called `MMCV_MODELS` that is imported from `MMCV`. Please check [mmselfsup/models/builder.py](https://github.com/open-mmlab/mmselfsup/blob/master/mmselfsup/models/builder.py) and refer to [mmcv/utils/registry.py](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) for more details.

- The `models` folder includes `algorithms`, `backbones`, `necks`, `heads`, `memories` and some required utils. The `algorithms` integrates the other main components to build the self-supervised learning algorithms, which is like `classifiers` in `MMCls` or `detectors` in `MMDet`.

- In OpenSelfSup, the names of `necks` are kind of confused and all in one file. Now, the `necks` are refactored, managed with one folder and renamed for easier understanding. Please check `mmselfsup/models/necks` for more details.

### Codebase Conventions

MMSelfSup renews codebase conventions as OpenSelfSup has not been updated for some time.

#### Configs

- MMSelfSup renames all config files to use new name convention. Please refer to [0_config](./tutorials/0_config.md) for more details.

- In the configs, some arguments of the class or names of the components have been modified.

  - One algorithm name has been modified: MOCO -> MoCo

  - As all models' components inherit `BaseModule` from `MMCV`, the models are initialized with `init_cfg`. Please use it to set your initialization. Besides, `init_weights` can also be used.

  - Please use new neck names to compose your algorithms, check it before write your own configs.

  - The normalization layers are all built with arguments `norm_cfg`.

#### Scripts

- The directory of `tools` is modified, thus it has more clear structure. It has several folders to manage different scripts. For example, it has two converter folders for models and data format. Besides, the benchmark related scripts are all in `benchmarks` folder, which has the same structure as `configs/benchmarks`.

- The arguments in `train.py` has been updated. Two major modifications are listed below.

  - Add `--cfg-options` to modify the config from cmd arguments.

  - Remove `--pretrained` and use `--cfg-options` to set the pretrained models.
