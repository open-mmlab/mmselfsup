# Get Started

- [Get Started](#get-started)
  - [Start to Train](#start-to-train)
    - [Train with a single GPU](#train-with-a-single-gpu)
    - [Train with CPU](#train-with-cpu)
    - [Train with multiple GPUs](#train-with-multiple-gpus)
    - [Train with multiple machines](#train-with-multiple-machines)
    - [Launch multiple jobs on a single machine](#launch-multiple-jobs-on-a-single-machine)
  - [Tools and Tips](#tools-and-tips)
    - [Count number of parameters](#count-number-of-parameters)
    - [Publish a model](#publish-a-model)
    - [Use t-SNE](#use-t-sne)
    - [Reproducibility](#reproducibility)

This page provides basic tutorials about the usage of MMSelfSup. For installation instructions, please see [install.md](install.md).

## Start to Train

**Note**: The default learning rate in config files is for specific number of GPUs, which is indicated in the config names. If using different number GPUs, the total batch size will change in proportion, you have to scale the learning rate following `new_lr = old_lr * new_ngpus / old_ngpus`.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

A simple example to start training:

```shell
python tools/train.py configs/selfsup/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k.py
```

### Train with CPU

```shell
export CUDA_VISIBLE_DEVICES=-1
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

**Note**: We do not recommend users to use CPU for training because it is too slow. We support this feature to allow users to debug on machines without GPU for convenience.

### Train with multiple GPUs

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```

Optional arguments:

- `--work-dir`: Inidicate your custom work directory to save checkpoints and logs.
- `--resume`: Automatically find the latest checkpoint in your work directory. Or set `--resume ${CHECKPOINT_PATH}` to load the specific checkpoint file.
- `--amp`: Enable automatic-mixed-precision training.
- `--cfg-options`: Setting `--cfg-options` will modify the original configs. For example, setting `--cfg-options randomness.seed=0` will set seed for random number.

An example to start training with 8 GPUs:

```shell
sh tools/dist_train.sh configs/selfsup/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k.py 8
```

Alternatively, if you run MMSelfSup on a cluster managed with **[slurm](https://slurm.schedmd.com/)**:

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} SRUN_ARGS=${SRUN_ARGS} sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [optional arguments]
```

An example to start training with 8 GPUs:

```shell
# The default setting: GPUS_PER_NODE=8 GPUS=8
sh tools/slurm_train.sh Dummy Test_job configs/selfsup/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k.py
```

### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG} ${GPUS}
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG} ${GPUS}
```

Usually it is slow if you do not have high speed networking like InfiniBand.

If you launch with **slurm**, the command is the same as that on single machine described above, but you need refer to [slurm_train.sh](https://github.com/open-mmlab/mmselfsup/blob/master/tools/slurm_train.sh) to set appropriate parameters and environment variables.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_1

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_2
```

If you use launch training jobs with slurm, you have two options to set different communication ports:

Option 1:

In `config1.py`:

```python
env_cfg = dict(dist_cfg=dict(backend='nccl', port=29500))
```

In `config2.py`:

```python
env_cfg = dict(dist_cfg=dict(backend='nccl', port=29501))
```

Then you can launch two jobs with config1.py and config2.py.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py [optional arguments]

CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py [optional arguments]
```

Option 2:

You can set different communication ports without the need to modify the configuration file, but have to set the `--cfg-options` to overwrite the default port in configuration file.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py --work-dir tmp_work_dir_1 --cfg-options env_cfg.dist_cfg.port=29500

CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py --work-dir tmp_work_dir_2 --cfg-options env_cfg.dist_cfg.port=29501
```

## Tools and Tips

### Count number of parameters

```shell
python tools/analysis_tools/count_parameters.py ${CONFIG_FILE}
```

### Publish a model

Before you publish a model, you may want to

- Convert model weights to CPU tensors.
- Delete the optimizer states.
- Compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

### Use t-SNE

We provide an off-the-shelf tool to visualize the quality of image representations by t-SNE.

Option 1:

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --work-dir ${WORK_DIR} [optional arguments]
```

Option 2:

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=${CHECKPOINT_PATH} \
    model.backbone.init_cfg.prefix='backbone' \
    [optional arguments]
```

Arguments of command:

- `CONFIG_FILE`: The config file for the pretrained model.
- `CHECKPOINT_PATH`: The path of model's checkpoint.
- `[optional arguments]`: As for optional arguments, you can refer to [visualize_tsne.py](https://github.com/open-mmlab/mmselfsup/blob/master/tools/analysis_tools/visualize_tsne.py)

### Reproducibility

If you want to make your performance exactly reproducible, please switch on `--cfg-options ramdonness.deterministic=True` to train the final model to be published. Note that this flag will switch off `torch.backends.cudnn.benchmark` and slow down the training speed.
