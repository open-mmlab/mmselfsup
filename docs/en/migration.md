# Differences between MMSelfSup and OpenSelfSup

This file records differences between the newest version of MMSelfSup with older versions and OpenSelfSup.

MMSelfSup goes through a refactoring and addresses many legacy issues. It is not compatitible with OpenSelfSup, i.e. the old config files are supposed to be updated as some arguments of the class or names of the components have been modified.

The major differences are in two folds: codebase conventions, modular design.

## Modular Design

In order to build more clear directory structure, MMSelfSup redesigns some of the modules.

### Datasets

- MMSelfSup merges some datasets to reduce some redundant codes.

  - Classification, Extraction, NPID -> OneViewDataset

  - BYOL, Contrastive -> MultiViewDataset

- The `data_sources` folder has been refactored, thus the loading function is more robust.

In addition, this part is still under refactoring, it will be released in following version.

### Models

- The registry mechanism is updated. Currently, the parts under the `models` folder are built with a parent called `MMCV_MODELS` that is imported from `MMCV`. Please check [mmselfsup/models/builder.py](https://github.com/open-mmlab/mmselfsup/blob/master/mmselfsup/models/builder.py) and refer to [mmcv/utils/registry.py](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) for more details.

- The `models` folder includes `algorithms`, `backbones`, `necks`, `heads`, `memories` and some required utils. The `algorithms` integrates the other main components to build the self-supervised learning algorithms, which is like `classifiers` in `MMCls` or `detectors` in `MMDet`.

- In OpenSelfSup, the names of `necks` are kind of confused and all in one file. Now, the `necks` are refactored, managed with one folder and renamed for easier understanding. Please check `mmselfsup/models/necks` for more details.

## Codebase Conventions

MMSelfSup renews codebase conventions as OpenSelfSup has not been updated for some time.

### Configs

- MMSelfSup renames all config files to use new name convention. Please refer to [0_config](./tutorials/0_config.md) for more details.

- In the configs, some arguments of the class or names of the components have been modified.

  - One algorithm name has been modified: MOCO -> MoCo

  - As all models' components inherit `BaseModule` from `MMCV`, the models are initialized with `init_cfg`. Please use it to set your initialization. Besides, `init_weights` can also be used.

  - Please use new neck names to compose your algorithms, check it before write your own configs.

  - The normalization layers are all built with arguments `norm_cfg`.

### Scripts

- The directory of `tools` is modified, thus it has more clear structure. It has several folders to manage different scripts. For example, it has two converter folders for models and data format. Besides, the benchmark related scripts are all in `benchmarks` folder, which has the same structure as `configs/benchmarks`.

- The arguments in `train.py` has been updated. Two major modifications are listed below.

  - Add `--cfg-options` to modify the config from cmd arguments.

  - Remove `--pretrained` and use `--cfg-options` to set the pretrained models.
