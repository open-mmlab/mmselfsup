# 分类

- [分类](#classification)
  - [VOC SVM/ Low-shot SVM](#voc-svm--low-shot-svm)
  - [线性评估和微调](#linear-evaluation-and-fine-tuning)
  - [ImageNet 半监督分类](#imagenet-semi-supervised-classification)
  - [ImageNet 最近邻分类](#imagenet-nearest-neighbor-classification)

在MMSelfSup中，我们为分类任务提供了许多基线，因此模型可以在不同分类任务上进行评估。这里有详细的教程和例子来阐述如何使用MMSelfSup来运行所有的分类基线。我们在`tools/benchmarks/classification/`文件夹中提供了所有的脚本，包含 2 个`.sh` 文件，一个用于与VOC SVM相关的分类任务，另一个用于ImageNet最近邻分类任务。

## VOC SVM / Low-shot SVM

为了运行这些基准，你首先应该准备好你的VOC数据集。请参考[prepare_data.md](./2_dataset_prepare.md)来获取数据准备的详细信息。

为了评估这些预训练的模型, 你可以运行如下指令。

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

此外，如果你想评估由runner保存的ckpt文件，你可以运行如下指令.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**为了使用ckpt进行测试，代码使用epoch\_\*.pth文件，这里不需要提取权重.**

注意：

- `${SELFSUP_CONFIG}`是自监督实验的配置文件.
- `${FEATURE_LIST}` 是一个字符串，用于指定从layer1到layer5的要评估特征；例如，如果你只想评估layer5，那么`FEATURE_LIST`是"feat5"，如果你想要评估所有的特征，那么`FEATURE_LIST`是"feat1 feat2 feat3 feat4 feat5" (用空格分隔)。如果为空，那么`FEATURE_LIST`默认是"feat5"。
- `PRETRAIN`：预训练模型文件。
- 如果你想改变GPU个数, 你可以在命令的前面加上`GPUS_PER_NODE=4 GPUS=4`。
- `EPOCH`是你想要测试的ckpt的轮数

## 线性评估和微调

线性评估和微调是最常见的两个基准。我们为线性评估和微调提供了配置文件和脚本来进行训练和测试。支持的数据集有 **ImageNet**，**Places205** 和 **iNaturalist18**。

首先，确保你已经安装[MIM](https://github.com/open-mmlab/mim)，这也是OpenMMLab的一个项目.

```shell
pip install openmim
```

此外，请参考MMMMClassification的[安装](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/install.md)和[数据准备](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/getting_started.md)。

然后运行如下命令。

```shell
# distributed version
bash tools/benchmarks/classification/mim_dist_train.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/mim_slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

注意：

- 默认的GPU数量是8。当改变GPU数量时，请同时改变配置文件中的`samples_per_gpu`参数来确保总的batch size是256。
- `CONFIG`：使用`configs/benchmarks/classification/`下的配置文件。具体来说，`imagenet` (除了`imagenet_*percent`文件), `places205` and `inaturalist2018`。
- `PRETRAIN`：预训练模型文件。

例子：

```shell
bash ./tools/benchmarks/classification/mim_dist_train.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/pretrained_model.pth
```

如果你想测试训练好的模型，请运行如下命令。

```shell
# distributed version
bash tools/benchmarks/classification/mim_dist_test.sh ${CONFIG} ${CHECKPOINT}

# slurm version
bash tools/benchmarks/classification//mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

注意：

- `CHECKPOINT`：你想测试的训练好的分类模型

例子：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/model.pth
```

## ImageNet半监督分类

为了运行ImageNet半监督分类，我们将使用和线性评估和微调一样的`.sh`脚本进行训练。

注意：

- 默认GPU数量是4.
- `CONFIG`：使用`configs/benchmarks/classification/imagenet/`下的配置文件，命名为`imagenet_*percent`的文件。
- `PRETRAIN`：预训练模型文件。

## ImageNet最近邻分类

```注意
仅支持CNN形式的主干网络 (例如ResNet50).
```

为评估用于ImageNet最近邻分类基准的预训练模型，你可以运行如下命令。

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh ${SELFSUP_CONFIG} ${PRETRAIN} [optional arguments]

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${CHECKPOINT} [optional arguments]
```

注意：

- `${SELFSUP_CONFIG}`是自监督实验的配置文件。
- `CHECKPOINT`：检查点模型文件的路径。
- 如果你想改变GPU的数量，你可以在命令的前面加上`GPUS_PER_NODE=4 GPUS=4`。
- `[optional arguments]`：用于可选参数，你可以参考这个[脚本](https://github.com/open-mmlab/mmselfsup/blob/1.x/tools/benchmarks/classification/knn_imagenet/test_knn.py)

命令的一个例子

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh \
    configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py \
    https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth
```
