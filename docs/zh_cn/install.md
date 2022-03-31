# 安装教程

## 依赖包

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 1.3.16+
- [mmcls](https://mmclassification.readthedocs.io/en/latest/install.html) 0.19.0+
- [mmdet](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) 2.16.0+
- [mmseg](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) 0.20.2+

下表显示了与 MMSelfSup 适配的 MMCV, MMClassification, MMDetection 和 MMSegmentation 的版本号。 为避免安装过程中出现问题，请参照下表安装适配的版本。

| MMSelfSup version |    MMCV version     |  MMClassification version  | MMSegmentation version | MMDetection version |
| :---------------: | :-----------------: | :------------------------: | :--------------------: | :-----------------: |
|  0.8.0 (master)   | mmcv-full >= 1.3.16 |      mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.7.1       | mmcv-full >= 1.3.16 | mmcls >= 0.19.0, <= 0.20.1 |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.6.0       | mmcv-full >= 1.3.16 |      mmcls >= 0.19.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.5.0       | mmcv-full >= 1.3.16 |             /              |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |


**注意:**
- 如果您已经安装了 mmcv, 您需要运行 `pip uninstall mmcv` 来卸载已经安装的 mmcv。 如果您在本地同时安装了 mmcv 和 mmcv-full, `ModuleNotFoundError` 将会抛出。
- 由于 MMSelfSup 从 MMClassification 引入了部分网络主干，所以您在使用 MMSelfSup 前必须安装 MMClassification。
- 如果您不需要 MMDetection 和 MMSegmentation 的基准评测，则安装它们不是必须的。

## 配置环境

1. 首先您需要用以下命令安装一个 conda 的虚拟环境，并激活它

    ```shell
    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    ```

2. 请参考 [官方教程](https://pytorch.org/) 安装 torch 和 torchvision, 例如您可以使用以下命令:

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    请确保您的 PyTorch 版本和 CUDA 版本匹配，具体您可以参考 [PyTorch 官网](https://pytorch.org/)。

    比如，您在 `/usr/local/cuda` 下安装了 CUDA 10.1，同时您想安装 PyTorch 1.7, 您可以使用以下命令安装适配 CUDA 10.1 的 PyTorch 预编译包。

    ```shell
    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch
    ```

    如果您选择从源编译 PyTorch 包，而不是选择预编译包，那么您在 CUDA 版本上拥有更多的选择，比如 9.0。


## 安装 MMSelfSup

1. 安装 MMCV 和 MMClassification

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    请将上面链接中 `{cu_version}` 和 `{torch_version}` 替换成您想要的版本。 比如, 安装最新版本 `mmcv-full`，同时适配 `CUDA 11.0` 和 `PyTorch 1.7.x`, 可以使用以下命令:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
    ```

    - PyTorch 在 1.x.0 和 1.x.1 之间通常是兼容的，故 mmcv-full 只提供 1.x.0 的编译包。如果您的 PyTorch 版本是 1.x.1，您可以放心地安装在 1.x.0 版本编译的 mmcv-full。

    您可以从 [这里](https://github.com/open-mmlab/mmcv#installation) 查找适配不同 PyTorch 和 CUDA 版本的 MMCV 版本。

    除此之外，您可以选择从源编译 MMCV，具体请参考 [MMCV安装文档](https://github.com/open-mmlab/mmcv#installation)。

    您可以使用以下命令安装 MMClassification：

    ```shell
    pip install mmcls
    ```

2. 克隆 MMSelfSup 并且安装

    ```shell
    git clone https://github.com/open-mmlab/mmselfsup.git
    cd mmselfsup
    pip install -v -e .
    ```

    **注意:**
    - 当您指定 `-e` 或 `develop`参数, MMSelfSup 采用开发者安装模式, 任何改动将会立即生效，而无需重新安装。

3. 安装 MMSegmentation 和 MMDetection

    您可以使用以下命令安装 MMSegmentation 和 MMDetection:

    ```shell
    pip install mmsegmentation mmdet
    ```

    除了使用 pip 安装 MMSegmentation 和 MMDetection, 您也可以使用 [mim](https://github.com/open-mmlab/mim), 例如:

    ```shell
    pip install openmim
    mim install mmdet
    mim install mmsegmentation
    ```

## 从零开始安装脚本

下面脚本提供了使用 conda 端到端安装 MMSelfSup 的所有命令。

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install -c pytorch pytorch torchvision -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# install mmdetection mmsegmentation
pip install mmsegmentation mmdet

git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
pip install -v -e .
```

## 另一种选择: 使用 Docker

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

## 安装校验

走完上面的步骤，为了确保您正确安装了 MMSelfSup 以及其各种依赖库，请使用下面脚本来完成校验：

```py
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

## 使用不同版本的 MMSelfSup

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
