# 语义分割

- [语义分割](#语义分割)
  - [训练](#训练)
  - [测试](#测试)

我们用　MMSegmentation　完成语义分割任务。首先请确保已安装OpenMMLab的　[MIM](https://github.com/open-mmlab/mim)　项目。

```shell
pip install openmim
mim install 'mmsegmentation>=1.0.0rc0'
```

下载和安装易如反掌。

此外，请参考 MMSegmentation 的[安装](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html) 和 [数据准备](https://mmsegmentation.readthedocs.io/en/dev-1.x/user_guides/2_dataset_prepare.html).

## 训练

安装成功后，您可以用简单的指令运行MMSeg。

```shell
# distributed 版本
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm 版本
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

注意:

- `CONFIG`: 用到 `configs/benchmarks/mmsegmentation/`　下的配置文件. 因为 OpenMMLab　的仓库支持用其他仓库的配置文件，我们可以像下面这样简单的从 MMSegmentation　中借用配置文件：

```shell
_base_ = 'mmseg::fcn/fcn_r50-d8_4xb2-40k_cityscapes-769x769.py'
```

我们也支持从零开始写您的配置文件。

- `PRETRAIN`: 预训练模型文件。
- `GPUS`: 您想用于训练的 GPU 数量。语义分割任务中默认用４块 GPU。

例如：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```

## 测试

训练完成后您可以用如下命令测试您的模型。

```shell
# distributed 版本
bash tools/benchmarks/mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm 版本
bash tools/benchmarks/mmsegmentation/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

注意:

- `CHECKPOINT`: 您想测试的训练好的语义分割模型。

实例如下：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```
