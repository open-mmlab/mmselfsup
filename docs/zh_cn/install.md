# 前提

在这一节中，我们展示了如何用 PyTorch 准备环境。

MMselfSup 可在 Linux 上运行 (Windows 和 macOS 平台不完全支持)。 要求 Python 3.6+, CUDA 9.2+ 和 PyTorch 1.5+。

如果您对 PyTorch 很熟悉，或者已经安装了它，可以忽略这部分并转到 [下一节](#%E5%AE%89%E8%A3%85)， 不然你可以按照下列步骤进行准备。

**Step 0.** 从 [官方网址](https://docs.conda.io/en/latest/miniconda.html) 下载并安装 Miniconda。

**Step 1.** 创建 conda 环境并激活

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** 按照 [官方教程](https://pytorch.org/get-started/locally/) 安装 PyTorch， 例如

GPU 平台:

```shell
conda install pytorch torchvision -c pytorch
```

CPU 平台:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们推荐用户按照我们的最优方案来安装 MMSelfSup，不过整体流程也可以是自定义的， 可参考 [自定义安装](#%E8%87%AA%E5%AE%9A%E4%B9%89%E5%AE%89%E8%A3%85) 章节

## 最优方案

**Step 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** 安装 MMSelfSup.

实例 a: 如果您直接或者开发 MMSelfSup， 从源安装:

```shell
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

实例 b: 如果您以 mmselfsup 为依赖项或者第三方库， 可使用 pip 安装：

```shell
pip install mmselfsup
```

## 安装校验

走完上面的步骤，为了确保您正确安装了 MMSelfSup 以及其各种依赖库，请使用下面脚本来完成校验：

```python
import torch

from mmselfsup.models import build_algorithm

model_config = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=-1),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=1000))

model = build_algorithm(model_config).cuda()

image = torch.randn((1, 3, 224, 224)).cuda()
label = torch.tensor([1]).cuda()

loss = model.forward_train(image, label)
```

如果您能顺利运行上面脚本，恭喜您已成功配置好所有环境。

## 自定义安装

### 基准测试

依照 [最优方案](#%E6%9C%80%E4%BC%98%E6%96%B9%E6%A1%88) 可以保证基本功能, 如果您需要一些下游任务来对您的预训练模型进行评测，例如检测或者分割， 请安装 [MMDetection](https://github.com/open-mmlab/mmdetection) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)。

如果您不运行 MMDetection 和 MMSegmentation 基准测试， 可以不进行安装。

您可以使用以下命令进行安装：

```shell
pip install mmdet mmsegmentation
```

若需要更详细的信息， 您可以参考 [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/get_started.md) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/get_started.md) 的安装指导页面。

### CUDA 版本

在安装 PyTorch 时， 您需要确认 CUDA 版本。 若您对此不清楚，可以按照我们的建议：

- 对于安培架构的 NVIDIA GPUs， 例如 GeForce 30 系列或者 NVIDIA A100, CUDA 11 是必须的。
- 对于较老版本的 NVIDIA GPUs， CUDA 11 是兼容的， 但是 CUDA 10.2 具有更好的兼容性以及更加轻量化。

请确认您的 GPU 驱动满足最小版本需求。 请参考 [此表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) 获取更多信息。

```{note}
如果您按照我们的最优方案安装 CUDA runtime 库是足够的，因为本地不会编译 CUDA 代码。但是如果您希望从源编译 MMCV 或开发其它 CUDA 算子， 您需要安装完整的 CUDA 工具包， 从 NVIDIA 的网站，https://developer.nvidia.com/cuda-downloads，并它的版本需要和 PyTorch 的 CUDA 版本相匹配。如准确的 cudatoolkit 版本 在 `conda install` 命令中。
```

### 不使用 MIM 安装 MMCV

MMCV 包含了 C++ 和 CUDA 扩展, 因此以一种复杂的方式依赖于 PyTorch。 MIM 自动解决了这种依赖关系，并使安装更加容易，然而，这不是必须的。

使用 pip 安装 MMCV， 而不是 MIM, 请参考 [MMCV 安装指南](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。 这需要根据 PyTorch 版本及其 CUDA 版本手动指定一个链接。

例如, 下列命令安装了 mmcv-full， 基于 PyTorch 1.10.x 和 CUDA 11.3。

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### 另一种选择: 使用 Docker

我们提供了一个配置好所有环境的 [Dockerfile](/docker/Dockerfile)。

```shell
# build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
docker build -f ./docker/Dockerfile --rm -t mmselfsup:torch1.10.0-cuda11.3-cudnn8 .
```

**重要:** 请确保您安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)。

运行下面命令:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmselfsup/data mmselfsup:torch1.10.0-cuda11.3-cudnn8 /bin/bash
```

`{DATA_DIR}` 是保存你所有数据集的根目录。

### 在 Google Colab 上安装

[Google Colab](https://research.google.com/) 一般已经安装了 PyTorch， 因此，我们只需要使用以下命令安装 MMCV 和 MMSeflSup。

**Step 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
!pip3 install openmim
!mim install mmcv-full
```

**Step 1.** 安装 MMSelfSup

```shell
!git clone https://github.com/open-mmlab/mmselfsup.git
%cd mmselfsup
!pip install -e .
```

**Step 2.** 安装校验

```python
import mmselfsup
print(mmselfsup.__version__)
# Example output: 0.9.0
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

## 问题解答

如果您在安装过程中遇到了什么问题， 请先查阅 [FAQ](faq.md) 页面.
您也可以在 GitHub [创建 issue](https://github.com/open-mmlab/mmselfsup/issues/new/choose)， 如果您没找到答案。

# 使用不同版本的 MMSelfSup

如果在您本地安装了多个版本的 MMSelfSup, 我们推荐您为这多个版本创建不同的虚拟环境。

另外一个方式就是在您程序的入口脚本处，插入以下代码片段 (`train.py`, `test.py` 或则其他任何程序入口脚本)

```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```

或则在不同版本的 MMSelfSup 的主目录中运行以下命令：

```shell
export PYTHONPATH="$(pwd)":$PYTHONPATH
```
