# Dummy MAE Wrapper

This is an example README for community `projects/`. We have provided detailed explanations for each field in the form of html comments, which are visible when you read the source of this README file. If you wish to submit your project to our main repository, then all the fields in this README are mandatory for others to understand what you have achieved in this implementation.

- [Dummy MAE Wrapper](#dummy-mae-wrapper)
  - [Description](#description)
  - [Usage](#usage)
    - [Setup Environment](#setup-environment)
    - [Data Preparation](#data-preparation)
    - [Pre-training Commands](#pre-training-commands)
      - [On Local Single GPU](#on-local-single-gpu)
      - [On Multiple GPUs](#on-multiple-gpus)
      - [On Multiple GPUs with Slurm](#on-multiple-gpus-with-slurm)
    - [Downstream Tasks Commands](#downstream-tasks-commands)
  - [Results](#results)
  - [Citation](#citation)
  - [Checklist](#checklist)

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

This project implements a dummy MAE wrapper, which prints "Welcome to MMSelfSup" during initialization.

## Usage

<!-- For a typical model, this section should contain the commands for dataset prepareation, pre-training, downstream tasks. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Setup Environment

Please refer to [Get Started](https://mmselfsup.readthedocs.io/en/1.x/get_started.html) documentation of MMSelfSup.

### Data Preparation

To show the dataset directory or provide the commands for dataset preparation if needed.

For example:

```text
data/
└── imagenet
    ├── train
    ├── val
    └── meta
        ├── train.txt
        └── val.txt
```

### Pre-training Commands

At first, you need to add the current folder to `PYTHONPATH`, so that Python can find your model files. In `example_project/` root directory, please run command below to add it.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Then run the following commands to train the model:

#### On Local Single GPU

```bash
mim train mmselfsup $CONFIG --work-dir $WORK_DIR

# a specific command example
mim train mmselfsup configs/dummy-mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py \
    --work-dir work_dirs/dummy_mae/
```

#### On Multiple GPUs

```bash
# a specific command examples, 8 GPUs here
mim train mmselfsup configs/dummy-mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py \
    --work-dir work_dirs/dummy_mae/ \
    --launcher pytorch --gpus 8
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints

#### On Multiple GPUs with Slurm

```bash
# a specific command example: 16 GPUs in 2 nodes
mim train mmselfsup configs/dummy-mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py \
    --work-dir work_dirs/dummy_mae/ \
    --launcher slurm --gpus 16 --gpus-per-node 8 \
    --partition $PARTITION
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- PARTITION: the slurm partition you are using

### Downstream Tasks Commands

In MMSelfSup's root directory, run the following command to train the downstream model:

```bash
mim train mmcls $CONFIG \
    --work-dir $WORK_DIR \
    --launcher pytorch -gpus 8 \
    [optional args]

# a specific command example
mim train mmcls configs/vit-base-p16_ft-8xb128-coslr-100e_in1k.py \
    --work-dir work_dirs/dummy_mae/classification/
    --launcher pytorch -gpus 8 \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth \
    model.backbone.init_cfg.prefix="backbone." \
    $PY_ARGS
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CHECKPOINT: the pretrained checkpoint of MMSelfSup saved in working directory, like `$WORK_DIR/epoch_300.pth`
- PY_ARGS: other optional args

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mae/README.md#models-and-benchmarks)
You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

If you have any downstream task results, you could list them here.

**For example:**

The Linear Eval and Fine-tuning results are based on ImageNet dataset.

<table class="docutils">
<thead>
  <tr>
	    <th>Algorithm</th>
	    <th>Backbone</th>
	    <th>Epoch</th>
      <th>Batch Size</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
  <tbody>
  <tr>
      <td>MAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>60.8</td>
      <td>83.1</td>
	</tr>
</tbody>
</table>

## Citation

<!-- You may remove this section if not applicable. -->

```bibtex
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}
```

## Checklist

Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress.

<!--The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `MMSelfSup.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Inference correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time feature vectors or losses matches that from the original codes. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [ ] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result. Due to the pretrain-downstream pipeline of self-supervised learning, this item requires at least one downstream result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/models/backbones/mae_vit.py) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/tests/test_models/test_backbones/test_mae_vit.py) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml and README.md

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mae/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mae/README.md) -->

- [ ] Refactor and Move your modules into the core package following the codebase's file hierarchy structure.
