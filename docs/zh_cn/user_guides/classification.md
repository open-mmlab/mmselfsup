# 分类

- [分类](#分类)
  - [VOC SVM / Low-shot SVM](#voc-svm--low-shot-svm)
  - [线性评估和微调](#线性评估和微调)
  - [ImageNet 半监督分类](#imagenet-半监督分类)
  - [ImageNet 最近邻分类](#imagenet-最近邻分类)

在 MMSelfSup 中，我们为分类任务提供了许多基线，因此模型可以在不同分类任务上进行评估。这里有详细的教程和例子来阐述如何使用 MMSelfSup 来运行所有的分类基线。我们在`tools/benchmarks/classification/`文件夹中提供了所有的脚本，包含 2 个`.sh` 文件，一个文件夹用于与 VOC SVM 相关的分类任务，另一个文件夹用于 ImageNet 最近邻分类任务。

## VOC SVM / Low-shot SVM

为了运行这些基准，您首先应该准备好您的 VOC 数据集。请参考 [prepare_data.md](./2_dataset_prepare.md) 来获取数据准备的详细信息。

为了评估这些预训练的模型, 您可以运行如下指令。

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

此外，如果您想评估由 runner 保存的ckpt文件，您可以运行如下指令。

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**使用 ckpt 进行测试，代码使用 epoch\_\*.pth 文件，这里不需要提取权重。**

备注：

- `${SELFSUP_CONFIG}` 是自监督实验的配置文件.
- `${FEATURE_LIST}` 是一个字符串，用于指定从 layer1 到 layer5 的要评估特征；例如，如果您只想评估 layer5，那么 `FEATURE_LIST` 是 "feat5"，如果您想要评估所有的特征，那么 `FEATURE_LIST` 是 "feat1 feat2 feat3 feat4 feat5" (用空格分隔)。如果为空，那么 `FEATURE_LIST` 默认是 "feat5"。
- `${PRETRAIN}`：预训练模型文件。
- 如果您想改变 GPU 个数, 您可以在命令的前面加上 `GPUS_PER_NODE=4 GPUS=4`。
- `${EPOCH}` 是您想要测试的 ckpt 的轮数

## 线性评估和微调

线性评估和微调是最常见的两个基准。我们为线性评估和微调提供了配置文件和脚本来进行训练和测试。支持的数据集有 **ImageNet**，**Places205** 和 **iNaturalist18**。

首先，确保您已经安装 [MIM](https://github.com/open-mmlab/mim)，这也是 OpenMMLab 的一个项目.

```shell
pip install openmim
```

此外，请参考 MMClassification 的[安装](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/install.md)和[数据准备](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/getting_started.md)。

然后运行如下命令。

```shell
# distributed version
bash tools/benchmarks/classification/mim_dist_train.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/mim_slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

备注：

- `${CONFIG}`：使用`configs/benchmarks/classification/`下的配置文件。具体来说，`imagenet` (除了`imagenet_*percent`文件), `places205` and `inaturalist2018`。
- `${PRETRAIN}`：预训练模型文件。

例子：

```shell
bash ./tools/benchmarks/classification/mim_dist_train.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/pretrained_model.pth
```

如果您想测试训练好的模型，请运行如下命令。

```shell
# distributed version
bash tools/benchmarks/classification/mim_dist_test.sh ${CONFIG} ${CHECKPOINT}

# slurm version
bash tools/benchmarks/classification//mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

备注：

- `${CHECKPOINT}`：您想测试的训练好的分类模型

例子：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/model.pth
```

## ImageNet 半监督分类

为了运行 ImageNet 半监督分类，我们将使用和线性评估和微调一样的`.sh`脚本进行训练。

备注：

- 默认GPU数量是4.
- `${CONFIG}`：使用`configs/benchmarks/classification/imagenet/`下的配置文件，命名为`imagenet_*percent`的文件。
- `${PRETRAIN}`：预训练模型文件。

## ImageNet 最近邻分类

```备注
仅支持 CNN 形式的主干网络 (例如 ResNet50)。
```

为评估用于 ImageNet 最近邻分类基准的预训练模型，您可以运行如下命令。

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh ${SELFSUP_CONFIG} ${PRETRAIN} [optional arguments]

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${CHECKPOINT} [optional arguments]
```

备注：

- `${SELFSUP_CONFIG}`：是自监督实验的配置文件。
- `${CHECKPOINT}`：检查点模型文件的路径。
- 如果您想改变GPU的数量，您可以在命令的前面加上`GPUS_PER_NODE=4 GPUS=4`。
- `[optional arguments]`：用于可选参数，您可以参考这个[脚本](https://github.com/open-mmlab/mmselfsup/blob/1.x/tools/benchmarks/classification/knn_imagenet/test_knn.py)

命令的一个例子

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh \
    configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py \
    https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth
```
