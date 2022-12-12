# Solution of FGIA ACCV 2022 (1st Place)

## Requirements

```shell
PyTorch 1.11.0
torchvision 0.12.0
CUDA 11.3
MMEngine >= 0.1.0
MMCV >= 2.0.0rc0
MMClassification >= 1.0.0rc1
```

## Preparing the dataset

First you should refactor the folder of your dataset in the following format:

```text
mmselfsup
|
|── data
|    |── WebiNat5000
|    |       |── meta
|    |       |    |── train.txt
|    |       |── train
|    |       |── testa
|    |       |── testb
```

The `train`, `testa`, and `testb` folders contain the same content with
those provided by the official website of the competition.

## Start pre-training

First, you should install all these requirements, following this [page](https://mmselfsup.readthedocs.io/en/dev-1.x/get_started.html).
Then change your current directory to the root of MMSelfSup

```shell
cd $MMSelfSup
```

Then you have the following two choices to start pre-training

### Slurm

If you have a cluster managed by Slurm, you can use the following command:

```shell
## we use 16 NVIDIA 80G A100 GPUs for pre-training
GPUS_PER_NODE=8 GPUS=16 SRUN_ARGS=${SRUN_ARGS} bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py [optional arguments]
```

### Pytorch

Or you can use the following two commands to start distributed training on two separate nodes:

```shell
# node 1
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} bash tools/dist_train.sh projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py 8
```

```shell
# node 2
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} bash tools/dist_train.sh projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py 8
```

All these logs and checkpoints will be saved under the folder `work_dirs`in the root.

Then you can use the pre-trained weights to initialize the model for downstream fine-tuning, following this [project](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/projects/fgia_accv2022_1st) in MMClassification.
