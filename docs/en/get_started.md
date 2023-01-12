# Get Started

- [Get Started](#get-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Best practices](#best-practices)
      - [Install from source](#install-from-source)
      - [Install as a Python package](#install-as-a-python-package)
    - [Verify the installation](#verify-the-installation)
    - [Customize installation](#customize-installation)
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

In this section, we demonstrate how to prepare an environment with PyTorch.

MMSelfSup works on Linux (Windows and macOS are not officially supported). It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.6+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the next Installation section. Otherwise, you can follow these steps for the preparation.
```

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

## Installation

We recommend users to follow our best practices to install MMSelfSup. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
```

**Step 1.** Install MMSelfSup.

According to your needs, we support two installation modes:

- [Install from source (Recommended)](#install-from-source): You want to develop your own self-supervised task or new features based on MMSelfSup framework, e.g., adding new datasets or models. And you can use all tools we provided.
- [Install as a Python package](#install-as-a-python-package): You just want to call MMSelfSup's APIs or import MMSelfSup's modules in your project.

#### Install from source

In this case, install mmselfsup from source:

```shell
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
git checkout 1.x
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Optionally, if you want to [contribute](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/docs/en/notes/contribution_guide.md) to MMSelfSup or experience experimental functions, please checkout to the `dev-1.x` branch:

```shell
git checkout dev-1.x
```

#### Install as a Python package

Just install with pip.

```shell
pip install 'mmselfsup>=1.0.0rc0'
```

### Verify the installation

To verify whether MMSelfSup is installed correctly, you can run the following command.

```python
import mmselfsup
print(mmselfsup.__version__)
# Example output: 1.0.0rc0 or newer
```

### Customize installation

#### Benchmark

The [Best practices](#best-practices) is for basic usage. If you need to evaluate your pre-trained model with some downstream tasks such as detection or segmentation, please also install [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

If you don't run MMDetection and MMSegmentation benchmarks, it is unnecessary to install them.

You can simply install MMDetection and MMSegmentation with the following command:

```shell
pip install 'mmdet>=3.0.0rc0' 'mmsegmentation>=1.0.0rc0'
```

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
