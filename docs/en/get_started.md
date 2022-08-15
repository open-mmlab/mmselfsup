# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMselfSup works on Linux (Windows and macOS are not officially supported). It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMSelfSup. However, the whole process is highly customizable. See [Customize Installation](#customized-installation) section for more information.

## Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMSelfSup.

Case a: If you develop and run mmselfsup directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmselfsup as a dependency or third-party package, install it with pip:

```shell
pip install mmselfsup
```

## Verify the installation

To verify whether MMSelfSup is installed correctly, we can run the following sample code to initialize a model and inference a demo image.

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

The above code is supposed to run successfully upon you finish the installation.

## Customized installation

### Benchmark

The [Best Practices](#best-practices) is for basic usage, if you need to evaluate your pre-training model with some downstream tasks such as detection or segmentation, please also install [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

If you don't run MMDetection and MMSegmentation benchmark, it is unnecessary to install them.

You can simply install MMDetection and MMSegmentation with the following command:

```shell
pip install mmdet mmsegmentation
```

For more details, you can check the installation page of [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md).

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Another option: Docker Image

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

### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMSeflSup with the following commands.

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmcv-full
```

**Step 1.** Install MMSelfSup from the source.

```shell
!git clone https://github.com/open-mmlab/mmselfsup.git
%cd mmselfsup
!pip install -e .
```

**Step 2.** Verification.

```python
import mmselfsup
print(mmselfsup.__version__)
# Example output: 0.9.0
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmselfsup/issues/new/choose) on GitHub if no solution is found.

# Using multiple MMSelfSup versions

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
