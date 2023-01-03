# 分类

- [分类](#分类)
  - [VOC SVM / Low-shot SVM](#voc-svm--low-shot-svm)
  - [线性评价与微调](#线性评价与微调)
  - [ImageNet 半监督学习分类](#imagenet-半监督学习分类)
  - [ImageNet 最近邻分类](#imagenet-最近邻分类)

在　MMSelfSup　中我们有很多分类基准测试，因此可以在不同分类工作上评估模型。以下是解释如何用　MMSelfSup　全部分类基准测试的详细教程和示例。
我们在 `tools/benchmarks/classification/` 文件夹提供脚本，在这个文件夹里有两个`.sh`文件，一个用于　VOC SVM　相关的分类任务，另一个用于 ImageNet 最近邻分类任务。

## VOC SVM / Low-shot SVM

运行这些基准测试之前请准备好 VOC 数据集。需要关于数据准备的详细信息的话请参考　[prepare_data.md](./2_dataset_prepare.md)。

您可以用如下命令评估预训练模型。

```shell
# distributed 版本
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm 版本
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

此外，如果您想评估运行者保存的 ckpt 文件，您可以用以下命令：

```shell
# distributed 版本
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm 版本
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**测试 ckpt 时代码用到 epoch\_\*.pth 文件, 没必要提取权重。**

提醒：

- `${SELFSUP_CONFIG}` 是自监督实验的配置文件。
- `${FEATURE_LIST}` 是一个用于具体说明要评估的来自第一层到第五层的特征的字符串。例如，如果您想只评估第五层，那么 `FEATURE_LIST` 是 "feat5"，如果您想评估所有层， `FEATURE_LIST` 是 "feat1 feat2 feat3 feat4 feat5" (用空格隔开). 如果为空, 默认 `FEATURE_LIST` 是 "feat5"。
- `PRETRAIN`: 预训练模型文件。
- 如果您想改变 GPU 数量，您可以在命令开头加上 `GPUS_PER_NODE=4 GPUS=4`。
- `EPOCH` 是您想测试的 ckpt 的轮数。

## 线性评价与微调

线性评价与微调是最常规的基准测试中的两个。我们提供用于线性评价与微调的启动训练和测试的配置文件与脚本文件。支持的数据集是 **ImageNet**, **Places205** 和 **iNaturalist18**。

首先，请确认您安装了OpenMMLab的 [MIM](https://github.com/open-mmlab/mim) 项目。

```shell
pip install openmim
```

此外，下载和数据集准备部分请参考 MMClassification 的 [installation](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/install.md) 和 [data preparation](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/getting_started.md).

然后请运行如下命令。

```shell
# distributed 版本
bash tools/benchmarks/classification/mim_dist_train.sh ${CONFIG} ${PRETRAIN}

# slurm 版本
bash tools/benchmarks/classification/mim_slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

注意:

- 默认 GPU 数量为 8。 改变 GPU 时，请对应的改变配置文件的 `samples_per_gpu` 以确保总批数为256。
- `CONFIG`: 用 `configs/benchmarks/classification/` 下的配置文件。具体来说，`imagenet` (除了 `imagenet_*percent` 文件夹), `places205` 和 `inaturalist2018`.
- `PRETRAIN`: 预训练模型文件.

例如:

```shell
bash ./tools/benchmarks/classification/mim_dist_train.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/pretrained_model.pth
```

如果您想测试训练好的模型，请运行以下命令。

```shell
# distributed 版本
bash tools/benchmarks/classification/mim_dist_test.sh ${CONFIG} ${CHECKPOINT}

# slurm 版本
bash tools/benchmarks/classification//mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

注意:

- `CHECKPOINT`: 您想测试的训练好的分类模型。

例如:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/model.pth
```

## ImageNet 半监督学习分类

为了运行 ImageNet 半监督学习分类，我们开始训练时仍用和线性评价与微调相同的 `.sh` 脚本文件

注意:

- 默认 GPU 数量为 4.
- `CONFIG`: 用 `configs/benchmarks/classification/imagenet/` 下名为 `imagenet_*percent` 的文件夹里的文件。
- `PRETRAIN`: 预训练模型文件。

## ImageNet 最近邻分类

```注意
只支持 CNN 风格的主干部分 (比如 ResNet50).
```

您可以运行如下命令来用最近邻基准测试评估预训练模型。

```shell
# distributed 版本
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh ${SELFSUP_CONFIG} ${PRETRAIN} [optional arguments]

# slurm 版本
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${CHECKPOINT} [optional arguments]
```

注意:

- `${SELFSUP_CONFIG}` 是自监督实验的配置文件。
- `CHECKPOINT`: 检查点模型文件的路径。
- 如果您想改变 GPU 数，您可以在命令开头加上 `GPUS_PER_NODE=4 GPUS=4`。
- `[optional arguments]`: 对于可选参数，您可以参考 [visualize_reconstruction.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/visualize_reconstruction.py)。

一个命令的例子

```shell
# distributed 版本
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh \
    configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py \
    https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth
```
