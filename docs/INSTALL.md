## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0/10.1
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2 (PyTorch-1.1 w/ NCCL-2.4.2 has a deadlock bug, see [here](https://github.com/open-mmlab/OpenSelfSup/issues/6))
- GCC(G++): 4.9/5.3/5.4/7.3

### Install openselfsup

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Install other third-party libraries.

```shell
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # optional for DeepCluster and ODC, assuming CUDA=10.0
```

d. Clone the openselfsup repository.

```shell
git clone https://github.com/open-mmlab/openselfsup.git
cd openselfsup
```

e. Install.

```shell
pip install -v -e .  # or "python setup.py develop"
```

f. Install Apex (optional), following the [official instructions](https://github.com/NVIDIA/apex), e.g.
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.

2. Following the above instructions, openselfsup is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.


### Prepare datasets

It is recommended to symlink your dataset root (assuming $YOUR_DATA_ROOT) to `$OPENSELFSUP/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

#### Prepare PASCAL VOC

Assuming that you usually store datasets in `$YOUR_DATA_ROOT` (e.g., for me, `/home/xhzhan/data/`).
This script will automatically download PASCAL VOC 2007 into `$YOUR_DATA_ROOT`, prepare the required files, create a folder `data` under `$OPENSELFSUP` and make a symlink `VOCdevkit`.

```shell
cd $OPENSELFSUP
bash tools/prepare_data/prepare_voc07_cls.sh $YOUR_DATA_ROOT
```

#### Prepare ImageNet and Places205

Taking ImageNet for example, you need to 1) download ImageNet; 2) create the following list files or download [here](https://drive.google.com/drive/folders/1wYkJU_1qRHEt1LPVjBiG6ddUFV-t9hVJ?usp=sharing) under $IMAGENET/meta/: `train.txt` and `val.txt` contains an image file name in each line, `train_labeled.txt` and `val_labeled.txt` contains `filename[space]label\n` in each line; `train_labeled_*percent.txt` are the down-sampled lists for semi-supervised evaluation. 3) create a symlink under `$OPENSELFSUP/data/`.

At last, the folder looks like:

```
OpenSelfSup
├── openselfsup
├── benchmarks
├── configs
├── data
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
│   ├── imagenet
│   │   ├── meta
│   │   |   ├── train.txt (for self-sup training, "filename\n" in each line)
│   │   |   ├── train_labeled.txt (for linear evaluation, "filename[space]label\n" in each line)
│   │   |   ├── train_labeled_1percent.txt (for semi-supervised evaluation)
│   │   |   ├── train_labeled_10percent.txt (for semi-supervised evaluation)
│   │   |   ├── val.txt
│   │   |   ├── val_labeled.txt (for evaluation)
│   │   ├── train
│   │   ├── val
│   ├── places205
│   │   ├── meta
│   │   |   ├── train.txt
│   │   |   ├── train_labeled.txt
│   │   |   ├── val.txt
│   │   |   ├── val_labeled.txt
│   │   ├── train
│   │   ├── val
```

### A from-scratch setup script

Here is a full script for setting up openselfsup with conda and link the dataset path. The script does not download ImageNet and Places datasets, you have to prepare them on your own.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install -c pytorch pytorch torchvision -y
git clone https://github.com/open-mmlab/OpenSelfSup.git
cd OpenSelfSup
pip install -v -e .

bash tools/prepare_data/prepare_voc07_cls.sh $YOUR_DATA_ROOT
ln -s $IMAGENET_ROOT data/imagenet
ln -s $PLACES_ROOT data/places205
```

### Using multiple OpenSelfSup versions

If there are more than one openselfsup on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```

Or run the following command in the terminal of corresponding folder to temporally use the current one.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Common Issues

1. The training hangs / deadlocks in some intermediate iteration. See this [issue](https://github.com/open-mmlab/OpenSelfSup/issues/6).
