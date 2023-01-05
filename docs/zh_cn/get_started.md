# 基础教程

- [基础教程](#基础教程)
<<<<<<< HEAD
  - [预备条件](#预备条件)
  - [安装](#安装)
    - [最佳实践](#最佳实践)
      - [从源代码安装](#从源代码安装)
      - [作为 Python 包安装](#作为-python-包安装)
    - [验证安装](#验证安装)
    - [自定义安装](#自定义安装)
      - [评测基准](#评测基准)
      - [CUDA 版本](#cuda-版本)
      - [在不使用 MIM 的情况下安装 MMEngine](#在不使用-mim-的情况下安装-mmengine)
      - [在不使用 MIM 的情况下安装 MMCV](#在不使用-mim-的情况下安装-mmcv)
      - [在仅有 CPU 的平台上安装](#在仅有-cpu-的平台上安装)
      - [在 Google Colab 上安装](#在-google-colab-上安装)
      - [通过 Docker 使用 MMSelfSup](#通过-docker-使用-mmselfsup)
    - [故障排除](#故障排除)
  - [使用多个 MMSelfSup 版本](#使用多个-mmselfsup-版本)
=======
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
>>>>>>> upstream/master

## 预备条件

在本节中，我们将演示如何使用 PyTorch 准备环境。

MMSelfSup 在 Linux 上运行（Windows 和 macOS 不受官方支持）。 它需要 Python 3.6+、CUDA 9.2+ 和 PyTorch 1.6+。

```{note}
如果您有使用 PyTorch 的经验并且已经安装了它，请跳过这一部分并跳到下一个安装环节。否则，您可以按照如下步骤进行准备。
```

**步骤 0.** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**步骤 1.** 创建一个 conda 环境并激活它。

```shell
<<<<<<< HEAD
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
=======
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${YOUR_WORK_DIR} [optional arguments]
>>>>>>> upstream/master
```

**步骤 2.** 按照[官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch，例如：

在 GPU 平台上：

```shell
<<<<<<< HEAD
conda install pytorch torchvision -c pytorch
=======
# checkpoints and logs saved in WORK_DIR=work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
bash tools/dist_train.sh configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py 8 --work_dir work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
>>>>>>> upstream/master
```

在 CPU 平台上：

```shell
<<<<<<< HEAD
conda install pytorch torchvision cpuonly -c pytorch
=======
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} SRUN_ARGS=${SRUN_ARGS} bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${YOUR_WORK_DIR} [optional arguments]
>>>>>>> upstream/master
```

## 安装

我们建议用户遵循我们的最佳实践来安装 MMSelfSup。 但是，整个过程是高度可定制的。 有关详细信息，请参阅[自定义安装](#customize-installation)部分。

### 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
<<<<<<< HEAD
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
=======
GPUS_PER_NODE=8 GPUS=8 bash tools/slurm_train.sh Dummy Test_job configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
>>>>>>> upstream/master
```

**步骤 1.** 安装 MMSelfSup。

根据您的需要，我们支持两种安装方式：

- [从源代码安装（推荐）](#从源代码安装): 您想开发自己的自监督任务或基于 MMSelfSup 框架的新功能，例如，添加新的数据集或模型。 您可以使用我们提供的所有工具。
- [作为 Python 包安装](#作为-python-包安装): 您只想在项目中调用 MMSelfSup 的 API 或导入 MMSelfSup 的模块。

#### 从源代码安装

在这种情况下，从源代码安装 MMSelfSup：

```shell
<<<<<<< HEAD
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
git checkout 1.x
pip install -v -e .
# "-v" 表示详细，或更多输出
# "-e" 表示以可编辑模式安装项目，
# 因此，对代码所做的任何本地修改都将生效，无需重新安装。
=======
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
>>>>>>> upstream/master
```

或者，如果您想为 MMSelfSup 做出[贡献](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/docs/zh_cn/notes/contribution_guides.md)或体验其正在实验中的功能，请查看 `dev-1.x` 分支：

```shell
<<<<<<< HEAD
git checkout dev-1.x
=======
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
>>>>>>> upstream/master
```

#### 作为 Python 包安装

直接用 pip 安装：

```shell
<<<<<<< HEAD
pip install 'mmselfsup>=1.0.0rc0'
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir tmp_work_dir_2
>>>>>>> upstream/master
```

### 验证安装

要验证是否正确安装了 MMSelfSup，可以运行以下命令：

```python
import mmselfsup
print(mmselfsup.__version__)
# 示例输出：1.0.0rc0 或更新版本
```

### 自定义安装

#### 评测基准

[最佳实践](#最佳实践)适用于基本用法。 如果您需要使用一些下游任务（例如检测或分割）来评估您的预训练模型，请同时安装 [MMDetection](https://github.com/open-mmlab/mmdetection) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)。

如果您不运行 MMDetection 和 MMSegmentation 基准测试，则无需安装它们。

您可以使用以下命令简单地安装 MMDetection 和 MMSegmentation：

```shell
pip install 'mmdet>=3.0.0rc0' 'mmsegmentation>=1.0.0rc0'
```

更多详细信息，您可以查看 [MMDetection](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/docs/zh_cn/get_started.md) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/zh_cn/get_started.md) 的安装页面。

#### CUDA 版本

安装 PyTorch 时，您需要指定 CUDA 的版本。 如果您不清楚选择哪个，请遵循我们的建议：

- 对于基于 Ampere 的 NVIDIA GPU，例如 GeForce 30 系列和 NVIDIA A100，CUDA 11 是必须的。
- 对于较旧的 NVIDIA GPU，CUDA 11 向后兼容，但 CUDA 10.2 提供更好的兼容性并且更轻量级。

请确保 GPU 驱动程序满足最低版本要求。有关详细信息，请参阅[此表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果您遵循我们的最佳实践，安装 CUDA 运行时库就足够了，因为不会在本地编译任何 CUDA 代码。 但是，如果您希望从源代码编译 MMCV 或开发其他 CUDA 算子，则需要从 NVIDIA 的[网站](https://developer.nvidia.com/cuda-downloads) 安装完整的 CUDA 工具包，其版本应与 PyTorch 的 CUDA 版本相匹配，即 `conda install` 命令中指定的 cudatoolkit 版本。
```

#### 在不使用 MIM 的情况下安装 MMEngine

想要使用 pip 而不是 MIM 安装 MMEngine，请遵循 [MMEngine 安装指南](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/get_started/installation.md)。

例如，您可以通过以下命令安装 MMEngine：

```shell
pip install mmengine
```

#### 在不使用 MIM 的情况下安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此以一种复杂的方式依赖于 PyTorch。 MIM 会自动解决此类依赖关系并使安装更容易。 但是，这不是必须的。

要使用 pip 而不是 MIM 安装 MMCV，请遵循 [MMCV 安装指南](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)。 这需要根据 PyTorch 版本及其 CUDA 版本手动指定 find-url。

例如，以下命令安装以 PyTorch 1.12.0 和 CUDA 11.6 构建的 mmcv-full。

```shell
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### 在仅有 CPU 的平台上安装

MMSelfSup 可以仅用于 CPU 环境。 在 CPU 模式下，您可以训练、测试或推断模型。

在这种模式下，一些功能会消失，通常是 GPU 编译的操作。 不过不用担心，MMSelfSup 中的几乎所有模型都不依赖这些操作。

#### 在 Google Colab 上安装

[Google Colab](https://research.google.com/) 通常会安装 PyTorch，因此我们只需要使用以下命令安装 MMCV 和 MMSeflSup。

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
!pip3 install openmim
!mim install mmengine
!mim install 'mmcv>=2.0.0rc1'
```

**步骤 1.** 从源代码安装 MMSelfSup。

```shell
!git clone https://github.com/open-mmlab/mmselfsup.git
%cd mmselfsup
!git checkout 1.x
!pip install -e .
```

**步骤 2.** 验证。

```python
import mmselfsup
print(mmselfsup.__version__)
# 示例输出：1.0.0rc0 或更新版本
```

```{note}
在 Jupyter 中，感叹号 `!` 用于调用外部可执行文件，而 `%cd` 是一个[魔法命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd ) 来更改 Python 的当前工作目录。
```

#### 通过 Docker 使用 MMSelfSup

我们提供了一个 [Dockerfile](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/docker/Dockerfile) 来构建镜像。请确保您的 [docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
<<<<<<< HEAD
# 使用 PyTorch 1.10.0、CUDA 11.3、CUDNN 8 构建镜像。
docker build -f ./docker/Dockerfile --rm -t mmselfsup:torch1.10.0-cuda11.3-cudnn8 .
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
>>>>>>> upstream/master
```

**重要提示：** 请确保您已安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

运行以下命令：

```shell
<<<<<<< HEAD
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmselfsup/data mmselfsup:torch1.10.0-cuda11.3-cudnn8 /bin/bash
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1 --cfg-options dist_params.port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2 --cfg-options dist_params.port=29501
>>>>>>> upstream/master
```

`{DATA_DIR}` 是包含所有这些数据集的本地文件夹。

### 故障排除

如果您在安装过程中遇到一些问题，请先查看[常见问题](notes/faq.md)页面。 如果没有找到解决方案，您可以在 GitHub 上[提交一个 issue](https://github.com/open-mmlab/mmselfsup/issues/new/choose)。

## 使用多个 MMSelfSup 版本

如果您的机器上有多个 mmselfsup，并且您想交替使用它们，推荐的方法是创建多个 conda 环境，并为不同的版本使用不同的环境。

另一种方法是将以下代码插入主脚本（train.py、test.py 或您运行的任何其他脚本）：

```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```

或者在对应根文件夹的终端中运行以下命令来暂时使用当前的版本：

```shell
export PYTHONPATH="$(pwd)":$PYTHONPATH
```
<<<<<<< HEAD
=======

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
>>>>>>> upstream/master
