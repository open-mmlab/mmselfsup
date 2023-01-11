# 分割

- [分割](#分割)
  - [训练](#训练)
  - [测试](#测试)

对于语义分割任务我们使用 MMSegmentation。首先确保您已经安装了 [MIM](https://github.com/open-mmlab/mim)，这也是 OpenMMLab 的一个项目。

```shell
pip install openmim
mim install 'mmsegmentation>=1.0.0rc0'
```

非常容易安装这个包。

此外，请参考 MMSegmentation 的[安装](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html)和[数据准备](https://mmsegmentation.readthedocs.io/en/dev-1.x/user_guides/2_dataset_prepare.html)。

## 训练

在安装完后，可以使用如下简单命令运行 MMSegmentation。

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

备注:

- `${CONFIG}`：使用`configs/benchmarks/mmsegmentation/`下的配置文件。由于 OpenMMLab 的算法库支持跨不同存储库引用配置文件，因此我们可以轻松使用 MMSegmentation 的配置文件，例如：

```shell
_base_ = 'mmseg::fcn/fcn_r50-d8_4xb2-40k_cityscapes-769x769.py'
```

从头开始写您的配置文件也是支持的。

- `${PARTITION}`：预训练模型文件
- `${GPUS}`: 您想用于训练的 GPU 数量，对于分割任务，我们默认采用 4 块 GPU。

例子：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```

## 测试

在训练之后，您可以运行如下命令测试您的模型。

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

备注：

- `${CHECKPOINT}`：您想测试的训练好的分割模型。

例子：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```
