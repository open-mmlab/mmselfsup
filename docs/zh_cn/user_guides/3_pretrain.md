# 教程 3: 使用现有模型进行预训练

- [教程 3: 使用现有模型进行预训练](#教程-3-使用现有模型进行预训练)
  - [使用单卡训练](#使用单卡训练)
  - [使用 CPU 训练](#使用-cpu-训练)
  - [使用多卡训练](#使用多卡训练)
  - [使用用多台机器训练](#使用用多台机器训练)
  - [在一台机器上启动多个任务](#在一台机器上启动多个任务)

本文档提供有关如何运行算法以及如何使用 MMSelfSup 中的一些工具的基本用法。有关安装说明和数据准备，请参阅 [install.md](install.md) 和 [prepare_data.md](prepare_data.md)。

##　开始训练

**注意**：配置文件中的默认学习率是针对特定数量的 GPU（GPU数量已在配置文件名称中注明）。如果使用不同数量的 GPUs，总的　batch size　将按比例变化，您必须按照 `new_lr = old_lr * new_ngpus / old_ngpus` 缩放学习率。

### 使用单卡训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

一个简单的例子来开启训练：

```shell
python tools/train.py configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py
```

### 使用 CPU 训练

```shell
export CUDA_VISIBLE_DEVICES=-1
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

**注意**。我们不建议用户使用CPU进行训练，因为它太慢了。我们支持这个功能，是为了方便用户在没有GPU的机器上进行调试。

### 使用多卡训练

```shell
sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```

可选参数：

- `--work-dir`：指示您的自定义工作目录以保存 checkpoints 和日志。
- `--resume`：自动在你的工作目录中查找最新的 checkpoints。或者设置 `--resume ${CHECKPOINT_PATH}` 来加载特定的 checkpoints 文件。
- `--amp`：启用自动混合精度训练。
- `--cfg-options`：设置 `--cfg-options` 将修改原始配置。例如，设置 `--cfg-options randomness.seed=0` 将为随机数设置种子。

使用 8 个 GPUs 开始训练的示例：

```shell
sh tools/dist_train.sh configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py 8
```

或者，如果您在使用 **[slurm](https://slurm.schedmd.com/)** 管理的集群上运行 MMSelfSup：

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} SRUN_ARGS=${SRUN_ARGS} sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} [optional arguments]
```

使用 8 个 GPUs 开始训练的示例：

```shell
# 默认设置: GPUS_PER_NODE=8 GPUS=8
sh tools/slurm_train.sh Dummy Test_job configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py
```

### 使用用多台机器训练

如果您想使用由以太网连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG} ${GPUS}
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG} ${GPUS}
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

如果您使用的是 **slurm** 来管理多台机器，您可以使用同在单台机器上一样的命令来启动任务，但是您必须得设置合适的环境变量和参数，具体可以参考 [slurm_train.sh](https://github.com/open-mmlab/mmselfsup/blob/master/tools/slurm_train.sh)。

### 在一台机器上启动多个任务

如果你在一台机器上启动多个任务，例如，在一台有 8 个 GPU 的机器上启动 2 个分别使用４块 GPU 的训练任务，你需要为每个任务指定不同的端口（默认为29500）以避免通信冲突。

如果你使用 `dist_train.sh` 来启动训练任务。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_1

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_2
```

如果你用 slurm 启动训练任务，你有两种方法来设置不同的通信端口。

**方法 1 :**

在 `config1.py` 中:

```python
env_cfg = dict(dist_cfg=dict(backend='nccl', port=29500))
```

在 `config2.py` 中:

```python
env_cfg = dict(dist_cfg=dict(backend='nccl', port=29501))
```

然后你可以用 config1.py 和 config2.py 启动两个任务。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py [optional arguments]

CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py [optional arguments]
```

**方法 2 :**

你可以设置不同的通信端口，而不需要修改配置文件，但必须设置 `--cfg-options` 来覆盖配置文件中的默认端口。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py --work-dir tmp_work_dir_1 --cfg-options env_cfg.dist_cfg.port=29500

CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py --work-dir tmp_work_dir_2 --cfg-options env_cfg.dist_cfg.port=29501
```
