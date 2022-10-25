# Get Started

- [Get Started](#get-started)
  - [Prerequisites](#prerequisites)
  - [安装](#安装)
    - [最佳实践](#最佳实践)
      - [从源代码安装](#从源代码安装)
      - [作为 Python 包安装](#作为-python-包安装)
    - [验证安装](#验证安装)
    - [自定义安装](#自定义安装)
      - [Benchmark](#benchmark)
      - [CUDA versions](#cuda-versions)
      - [Install MMEngine without MIM](#install-mmengine-without-mim)
      - [Install MMCV without MIM](#install-mmcv-without-mim)
      - [Install on CPU-only platforms](#install-on-cpu-only-platforms)
      - [Install on Google Colab](#install-on-google-colab)
      - [Using MMSelfSup with Docker](#using-mmselfsup-with-docker)
    - [Trouble shooting](#trouble-shooting)
  - [Using Multiple MMSelfSup Versions](#using-multiple-mmselfsup-versions)

## Prerequisites

在本节中，我们将演示如何使用 PyTorch 准备环境。

MMSelfSup 在 Linux 上运行（Windows 和 macOS 不受官方支持）。 它需要 Python 3.6+、CUDA 9.2+ 和 PyTorch 1.6+。

```{note}
如果您有使用 PyTorch 的经验并且已经安装了它，请跳过这一部分并跳到下一个安装环节。 否则，您可以按照如下步骤进行准备。
```

**步骤 0.** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**步骤 1.** 创建一个 conda 环境并激活它。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**步骤 2.** 按照[官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch，例如：

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## 安装

我们建议用户遵循我们的最佳实践来安装 MMSelfSup。 但是，整个过程是高度可定制的。 有关详细信息，请参阅[自定义安装](#customize-installation)部分。

### 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
```

**步骤 1.** 安装 MMSelfSup.

根据您的需要，我们支持两种安装方式：

- [从源代码安装（推荐）](#install-from-source): 您想开发自己的自监督任务或基于 MMSelfSup 框架的新功能，例如，添加新的数据集或模型。 您可以使用我们提供的所有工具。
- [作为 Python 包安装](#install-as-a-python-package): 您只想在项目中调用 MMSelfSup 的 API 或导入 MMSelfSup 的模块。

#### 从源代码安装

在这种情况下，从源代码安装 mmselfsup：

```shell
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
git checkout 1.x
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

或者，如果您想为 MMSelfSup 做出贡献或体验其实验中的功能，请查看 `dev-1.x` 分支：

```shell
git checkout dev-1.x
```

#### 作为 Python 包安装

直接用 pip 安装：

```shell
pip install 'mmselfsup>=1.0.0rc0'
```

### 验证安装

要验证是否正确安装了 MMSelfSup，可以运行以下命令：

```python
import mmselfsup
print(mmselfsup.__version__)
# Example output: 1.0.0rc0 or newer
```

我们还提供以下示例代码来初始化模型并推断演示图像：

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

上述代码应该在您完成安装后成功运行。

### 自定义安装

#### 基准

[最佳实践](#best-practices)适用于基本用法。 如果您需要使用一些下游任务（例如检测或分割）来评估您的预训练模型，请同时安装 [MMDetection](https://github.com/open-mmlab/mmdetection) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)。

如果您不运行 MMDetection 和 MMSegmentation 基准测试，则无需安装它们。

您可以使用以下命令简单地安装 MMDetection 和 MMSegmentation：

```shell
pip install 'mmdet>=3.0.0rc0' 'mmsegmentation>=1.0.0rc0'
```

更多详细信息，您可以查看 [MMDetection](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/docs/en/get_started.md) 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/get_started.md)) 的安装页面。

For more details, you can check the installation page of [MMDetection](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/docs/en/get_started.md) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/get_started.md).

#### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

#### Install MMEngine without MIM

To install MMEngine with pip instead of MIM, please follow [MMEngine installation guides](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/get_started/installation.md).

For example, you can install MMEngine by the following command.

```shell
pip install mmengine
```

#### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command installs mmcv-full built for PyTorch 1.12.0 and CUDA 11.6.

```shell
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### Install on CPU-only platforms

MMSelfSup can be built for CPU only environment. In CPU mode, you can train, test or inference a model.

Some functionalities are gone in this mode, usually GPU-compiled ops. But don't
worry, almost all models in MMSelfSup don't depend on these ops.

#### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMSeflSup with the following commands.

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmengine
!mim install 'mmcv>=2.0.0rc1'
```

**Step 1.** Install MMSelfSup from the source.

```shell
!git clone https://github.com/open-mmlab/mmselfsup.git
%cd mmselfsup
!git checkout 1.x
!pip install -e .
```

**Step 2.** Verification.

```python
import mmselfsup
print(mmselfsup.__version__)
# Example output: 1.0.0rc0 or newer
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

#### Using MMSelfSup with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.10.0, CUDA 11.3, CUDNN 8.
docker build -f ./docker/Dockerfile --rm -t mmselfsup:torch1.10.0-cuda11.3-cudnn8 .
```

**Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Run the following command:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmselfsup/data mmselfsup:torch1.10.0-cuda11.3-cudnn8 /bin/bash
```

`{DATA_DIR}` is your local folder containing all these datasets.

### Trouble shooting

If you have some issues during the installation, please first view the [FAQ](notes/faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmselfsup/issues/new/choose) on GitHub if no solution is found.

## Using Multiple MMSelfSup Versions

If there are more than one mmselfsup on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)

```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```

Or run the following command in the terminal of corresponding root folder to temporally use the current one.

```shell
export PYTHONPATH="$(pwd)":$PYTHONPATH
```
