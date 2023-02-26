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
