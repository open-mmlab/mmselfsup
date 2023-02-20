# GreenMIM Pre-training Model

- [GreenMIM Pre-training Model](#maskfeat-pre-training-with-video)
  - [Description](#description)
  - [Usage](#usage)
    - [Setup Environment](#setup-environment)
    - [Data Preparation](#data-preparation)
    - [Pre-training Commands](#pre-training-commands)
      - [On Local Single GPU](#on-local-single-gpu)
      - [On Multiple GPUs](#on-multiple-gpus)
      - [On Multiple GPUs with Slurm](#on-multiple-gpus-with-slurm)
  - [Citation](#citation)
  - [Checklist](#checklist)

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

Author: @xfguo-ucas

This is the implementation of **GreenMIM** with ImageNet.

## Usage

<!-- For a typical model, this section should contain the commands for dataset prepareation, pre-training, downstream tasks. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Setup Environment

Requirements:

- MMSelfSup >= 1.0.0rc7

Please refer to [Get Started](https://mmselfsup.readthedocs.io/en/1.x/get_started.html) documentation of MMSelfSup to finish installation.

### Data Preparation

You can refer to the [documentation](https://mmclassification.readthedocs.io/en/latest/getting_started.html) in mmcls.

### Pre-training Commands

At first, you need to add the current folder to `PYTHONPATH`, so that Python can find your model files. In `projects/greenmim/` root directory, please run command below to add it.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Then run the following commands to train the model:

#### On Local Single GPU

```bash
# train with mim
mim train mmselfsup ${CONFIG} --work-dir ${WORK_DIR}

# a specific command example
mim train mmselfsup configs/greenmim_swin-base_16xb128-amp-coslr-100e_in1k.py \
    --work-dir work_dirs/selfsup/greenmim_swin-base_16xb128-amp-coslr-100e_in1k/

# train with scripts
python tools/train.py configs/greenmim_swin-base_16xb128-amp-coslr-100e_in1k.py \
    --work-dir work_dirs/selfsup/greenmim_swin-base_16xb128-amp-coslr-100e_in1k/
```

#### On Multiple GPUs

```bash
# train with mim
# a specific command examples, 8 GPUs here
mim train mmselfsup configs/greenmim_swin-base_16xb128-amp-coslr-100e_in1k.py \
    --work-dir work_dirs/selfsup/greenmim_swin-base_16xb128-amp-coslr-100e_in1k/ \
    --launcher pytorch --gpus 8

# train with scripts
bash tools/dist_train.sh configs/greenmim_swin-base_16xb128-amp-coslr-100e_in1k.py 8
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints

#### On Multiple GPUs with Slurm

```bash
# train with mim
mim train mmselfsup configs/greenmim_swin-base_16xb128-amp-coslr-100e_in1k.py \
    --work-dir work_dirs/selfsup/greenmim_swin-base_16xb128-amp-coslr-100e_in1k/ \
    --launcher slurm --gpus 16 --gpus-per-node 8 \
    --partition ${PARTITION}

# train with scripts
GPUS_PER_NODE=8 GPUS=16 bash tools/slurm_train.sh ${PARTITION} greenmim \
    configs/greenmim_swin-base_16xb128-amp-coslr-100e_in1k.py \
    --work-dir work_dirs/selfsup/greenmim_swin-base_16xb128-amp-coslr-100e_in1k/
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- PARTITION: the slurm partition you are using

## Citation

```bibtex
@article{huang2022green,
  title={Green Hierarchical Vision Transformer for Masked Image Modeling},
  author={Huang, Lang and You, Shan and Zheng, Mingkai and Wang, Fei and Qian, Chen and Yamasaki, Toshihiko},
  journal={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022}
}
```

## Checklist

Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress.

<!--The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `MMSelfSup.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Inference correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time feature vectors or losses matches that from the original codes. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result. Due to the pretrain-downstream pipeline of self-supervised learning, this item requires at least one downstream result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/models/backbones/mae_vit.py) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/tests/test_models/test_backbones/test_mae_vit.py) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mae/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mae/README.md) -->

- [ ] Refactor and Move your modules into the core package following the codebase's file hierarchy structure.
