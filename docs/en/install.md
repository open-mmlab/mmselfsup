# Installation

## Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.5+
- CUDA 9.2+
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) 1.3.16+
- [mmcls](https://mmclassification.readthedocs.io/en/latest/install.html) 0.19.0+
- [mmdet](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) 2.16.0+
- [mmseg](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) 0.20.2+

Compatible MMCV, MMClassification, MMDetection and MMSegmentation versions are shown below. Please install the correct version of them to avoid installation issues.

| MMSelfSup version |    MMCV version     |  MMClassification version  | MMSegmentation version | MMDetection version |
| :---------------: | :-----------------: | :------------------------: | :--------------------: | :-----------------: |
|  0.8.0 (master)   | mmcv-full >= 1.3.16 |      mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.7.1       | mmcv-full >= 1.3.16 | mmcls >= 0.19.0, <= 0.20.1 |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.6.0       | mmcv-full >= 1.3.16 |      mmcls >= 0.19.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.5.0       | mmcv-full >= 1.3.16 |             /              |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |

**Note:**

- You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
- As MMSelfSup imports some backbones from MMClassification, you need to install MMClassification before using MMSelfSup.
- If you don't run MMDetection and MMSegmentation benchmark, it is unnecessary to install them.

## Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install PyTorch 1.7, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt package, you can use more CUDA versions such as 9.0.


## Install MMSelfSup

1. Install MMCV and MMClassification

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11.0` and `PyTorch 1.7.x`, use the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
    ```

    - mmcv-full is only compiled on PyTorch 1.x.0 because the compatibility usually holds between 1.x.0 and 1.x.1. If your PyTorch version is 1.x.1, you can install mmcv-full compiled with PyTorch 1.x.0 and it usually works well.

    See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

    Optionally you can compile mmcv from source if you need to develop both mmcv and mmselfsup. Refer to the [guide](https://github.com/open-mmlab/mmcv#installation) for details.

    You can simply install MMClassification with the following command:

    ```shell
    pip install mmcls
    ```

2. Clone MMSelfSup repository and install

    ```shell
    git clone https://github.com/open-mmlab/mmselfsup.git
    cd mmselfsup
    pip install -v -e .
    ```

    **Note:**
    - When specifying `-e` or `develop`, MMSelfSup is installed on dev mode, any local modifications made to the code will take effect without reinstallation.

3. Install MMSegmentation and MMDetection

    You can simply install MMSegmentation and MMDetection with the following command:

    ```shell
    pip install mmsegmentation mmdet
    ```

    In addition to installing MMSegmentation and MMDetection by pip, user can also install them by [mim](https://github.com/open-mmlab/mim).

    ```shell
    pip install openmim
    mim install mmdet
    mim install mmsegmentation
    ```

## A from-scratch setup script

Here is a full script for setting up mmselfsup with conda.

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

## Another option: Docker Image

We provide a [Dockerfile](/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
docker build -f ./docker/Dockerfile --rm -t mmselfsup:torch1.10.0-cuda11.3-cudnn8 .
```

**Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Run the following cmd:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmselfsup/data mmselfsup:torch1.10.0-cuda11.3-cudnn8 /bin/bash
```

`{DATA_DIR}` is your local folder containing all these datasets.

## Verification

To verify whether MMSelfSup is installed correctly, we can run the following sample code to initialize a model and inference a demo image.

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

The above code is supposed to run successfully upon you finish the installation.

## Using multiple MMSelfSup versions

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
