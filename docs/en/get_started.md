# Get Started

<<<<<<< HEAD
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
=======
- [Getting Started](#getting-started)
  - [Train existing methods](#train-existing-methods)
    - [Training with CPU](#training-with-cpu)
    - [Train with single/multiple GPUs](#train-with-singlemultiple-gpus)
    - [Train with multiple machines](#train-with-multiple-machines)
    - [Launch multiple jobs on a single machine](#launch-multiple-jobs-on-a-single-machine)
  - [Benchmarks](#benchmarks)
  - [Tools and Tips](#tools-and-tips)
    - [Count number of parameters](#count-number-of-parameters)
    - [Publish a model](#publish-a-model)
    - [Use t-SNE](#use-t-sne)
    - [MAE Visualization](#mae-visualization)
    - [Reproducibility](#reproducibility)
>>>>>>> upstream/master

## Prerequisites

In this section, we demonstrate how to prepare an environment with PyTorch.

MMSelfSup works on Linux (Windows and macOS are not officially supported). It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.6+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the next Installation section. Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
<<<<<<< HEAD
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
=======
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${YOUR_WORK_DIR} [optional arguments]
>>>>>>> upstream/master
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
<<<<<<< HEAD
conda install pytorch torchvision -c pytorch
=======
# checkpoints and logs saved in WORK_DIR=work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
bash tools/dist_train.sh configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py 8 --work-dir work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
>>>>>>> upstream/master
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
<<<<<<< HEAD
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
=======
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} SRUN_ARGS=${SRUN_ARGS} bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${YOUR_WORK_DIR} [optional arguments]
>>>>>>> upstream/master
```

**Step 1.** Install MMSelfSup.

According to your needs, we support two installation modes:

- [Install from source (Recommended)](#install-from-source): You want to develop your own self-supervised task or new features based on MMSelfSup framework, e.g., adding new datasets or models. And you can use all tools we provided.
- [Install as a Python package](#install-as-a-python-package): You just want to call MMSelfSup's APIs or import MMSelfSup's modules in your project.

#### Install from source

In this case, install mmselfsup from source:

```shell
<<<<<<< HEAD
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup
git checkout 1.x
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
=======
GPUS_PER_NODE=8 GPUS=8 bash tools/slurm_train.sh Dummy Test_job configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py work_dirs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k/
>>>>>>> upstream/master
```

Optionally, if you want to [contribute](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/docs/en/notes/contribution_guide.md) to MMSelfSup or experience experimental functions, please checkout to the `dev-1.x` branch:

```shell
<<<<<<< HEAD
git checkout dev-1.x
=======
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
>>>>>>> upstream/master
```

#### Install as a Python package

Just install with pip.

```shell
<<<<<<< HEAD
pip install 'mmselfsup>=1.0.0rc0'
=======
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
>>>>>>> upstream/master
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
<<<<<<< HEAD
# build an image with PyTorch 1.10.0, CUDA 11.3, CUDNN 8.
docker build -f ./docker/Dockerfile --rm -t mmselfsup:torch1.10.0-cuda11.3-cudnn8 .
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2
>>>>>>> upstream/master
```

**Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Run the following command:

```shell
<<<<<<< HEAD
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmselfsup/data mmselfsup:torch1.10.0-cuda11.3-cudnn8 /bin/bash
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py tmp_work_dir_1 --cfg-options dist_params.port=29500
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py tmp_work_dir_2 --cfg-options dist_params.port=29501
>>>>>>> upstream/master
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
<<<<<<< HEAD
=======

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

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --work-dir ${WORK_DIR} [optional arguments]
```

Arguments:

- `CONFIG_FILE`: config file for the pre-trained model.
- `CKPT_PATH`: the path of model's checkpoint.
- `WORK_DIR`: the directory to save the results of visualization.
- `[optional arguments]`: for optional arguments, you can refer to [visualize_tsne.py](https://github.com/open-mmlab/mmselfsup/blob/master/tools/analysis_tools/visualize_tsne.py)

### MAE Visualization

We provide a tool to visualize the mask and reconstruction image of MAE model.

```shell
python tools/misc/mae_visualization.py ${IMG_PATH} ${CONFIG_FILE} ${CKPT_PATH} ${OUT_FILE} --device ${DEVICE}
```

Arguments:

- `IMG_PATH`: an image path used for visualization.
- `CONFIG_FILE`: config file for the pre-trained model.
- `CKPT_PATH`: the path of model's checkpoint.
- `OUT_FILE`: the image path used for visualization results.
- `DEVICE`: device used for inference.

An example:

```shell
python tools/misc/mae_visualization.py tests/data/color.jpg configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py mae_epoch_400.pth results.jpg --device 'cuda:0'
```

### Reproducibility

If you want to make your performance exactly reproducible, please switch on `--deterministic` to train the final model to be published. Note that this flag will switch off `torch.backends.cudnn.benchmark` and slow down the training speed.
>>>>>>> upstream/master
