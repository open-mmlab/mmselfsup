# 基础教程

- [基础教程](#基础教程)
  - [训练已有的算法](#训练已有的算法)
    - [使用 CPU 训练](#使用-cpu-训练)
    - [使用 单张/多张 显卡训练](#使用-单张多张-显卡训练)
    - [使用多台机器训练](#使用多台机器训练)
    - [在一台机器上启动多个任务](#在一台机器上启动多个任务)
  - [基准测试](#基准测试)
  - [工具和建议](#工具和建议)
    - [统计模型的参数](#统计模型的参数)
    - [发布模型](#发布模型)
    - [使用 t-SNE 来做模型可视化](#使用-t-sne-来做模型可视化)
    - [MAE 可视化](#mae-可视化)
    - [可复现性](#可复现性)

本文档提供 MMSelfSup 相关用法的基础教程。 如果您对如何安装 MMSelfSup 以及其相关依赖库有疑问, 请参考 [安装文档](install.md).

## 训练已有的算法

**注意**: 当您启动一个任务的时候，默认会使用8块显卡. 如果您想使用少于或多余8块显卡, 那么你的 batch size 也会同比例缩放，同时您的学习率服从一个线性缩放原则, 那么您可以使用以下公式来调整您的学习率: `new_lr = old_lr * new_ngpus / old_ngpus`. 除此之外，我们推荐您使用 `tools/dist_train.sh` 来启动训练任务，即便您只使用一块显卡, 因为 MMSelfSup 中有些算法不支持非分布式训练。

### 使用 CPU 训练

```shell
export CUDA_VISIBLE_DEVICES=-1
python tools/train.py ${CONFIG_FILE}
```

**注意**: 我们不推荐用户使用 CPU 进行训练， 因为 CPU 的训练速度很慢，一些算法仅支持分布式训练, 例如 `SyncBN`，该方法需要分布式进行训练，我们支持这个功能是为了方便用户在没有 GPU 的机器上进行调试。

### 使用 单张/多张 显卡训练

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${YOUR_WORK_DIR} [optional arguments]
```

可选参数:

- `--resume-from ${CHECKPOINT_FILE}`: 从某个 checkpoint 处继续训练.
- `--deterministic`: 开启 "deterministic" 模式, 虽然开启会使得训练速度降低，但是会保证结果可复现。

例如:

```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
bash tools/dist_train.sh configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py 8 --work_dir work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
```

**注意**: 在训练过程中, checkpoints 和 logs 被保存在同一目录层级下.

此外, 如果您在一个被 [slurm](https://slurm.schedmd.com/) 管理的集群中训练， 您可以使用以下的脚本开展训练:

```shell
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} SRUN_ARGS=${SRUN_ARGS} bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${YOUR_WORK_DIR} [optional arguments]
```

例如:

```shell
GPUS_PER_NODE=8 GPUS=8 bash tools/slurm_train.sh Dummy Test_job configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
```

### 使用多台机器训练

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

如果您使用的是 slurm 来管理多台机器，您可以使用同在单台机器上一样的命令来启动任务，但是您必须得设置合适的环境变量和参数，具体可以参考[slurm_train.sh](../../tools/slurm_train.sh)。

### 在一台机器上启动多个任务

如果您想在一台机器上启动多个任务，比如说，您启动两个4卡的任务在一台8卡的机器上，您需要为每个任务指定不懂的端口来防止端口冲突。

如果您使用  `dist_train.sh`  来启动训练任务:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_2
```

如果您使用 slurm 来启动训练任务，你有两种方式来为每个任务设置不同的端口:

方法 1:

在 `config1.py` 中, 做如下修改:

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py`中，做如下修改:

```python
dist_params = dict(backend='nccl', port=29501)
```

然后您可以通过 config1.py 和 config2.py 来启动两个不同的任务.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
```

方法 2:

除了修改配置文件之外, 您可以设置 `cfg-options` 来重写默认的端口号:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1 --cfg-options dist_params.port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2 --cfg-options dist_params.port=29501
```

## 基准测试

我们同时提供多种命令来评估您的预训练模型, 具体您可以参考[Benchmarks](./tutorials/6_benchmarks.md)。

## 工具和建议

### 统计模型的参数

```shell
python tools/analysis_tools/count_parameters.py ${CONFIG_FILE}
```

### 发布模型

当你发布一个模型之前，您可能想做以下几件事情

- 将模型的参数转为 CPU tensor.
- 删除 optimizer 的状态参数.
- 计算 checkpoint 文件的哈希值，并将其添加到 checkpoint 的文件名中.

您可以使用以下命令来完整上面几件事情:

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

### 使用 t-SNE 来做模型可视化

我们提供了一个开箱即用的来做图片向量可视化的方法:

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --work-dir ${WORK_DIR} [optional arguments]
```

参数:

- `CONFIG_FILE`: 训练预训练模型的参数配置文件.
- `CKPT_PATH`: 预训练模型的路径.
- `WORK_DIR`: 保存可视化结果的路径.
- `[optional arguments]`: 可选参数，具体可以参考 [visualize_tsne.py](../../tools/analysis_tools/visualize_tsne.py)

### MAE 可视化

我们提供了一个对 MAE 掩码效果和重建效果可视化可视化的方法:

```shell
python tools/misc/mae_visualization.py ${IMG_PATH} ${CONFIG_FILE} ${CKPT_PATH} ${OUT_FILE} --device ${DEVICE}
```

参数:

- `IMG_PATH`: 用于可视化的图片
- `CONFIG_FILE`: 训练预训练模型的参数配置文件.
- `CKPT_PATH`: 预训练模型的路径.
- `OUT_FILE`: 用于保存可视化结果的图片路径
- `DEVICE`: 用于推理的设备.

示例:

```shell
python tools/misc/mae_visualization.py tests/data/color.jpg configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py mae_epoch_400.pth results.jpg --device 'cuda:0'
```

### 可复现性

如果您想确保模型精度的可复现性，您可以设置 `--deterministic` 参数。但是，开启 `--deterministic` 意味着关闭 `torch.backends.cudnn.benchmark`, 所以会使模型的训练速度变慢。
