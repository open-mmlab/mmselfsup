# 教程 6：运行基准评测

在 MMSelfSup 中，我们提供了许多基准评测，因此模型可以在不同的下游任务中进行评估。这里提供了全面的教程和例子来解释如何用 MMSelfSup 运行所有的基准。

- [教程 6：运行基准评测](#%E6%95%99%E7%A8%8B6%EF%BC%9A%E8%BF%90%E8%A1%8C%E5%9F%BA%E5%87%86%E8%AF%84%E6%B5%8B)
  - [分类](#%E5%88%86%E7%B1%BB)
    - [VOC SVM / Low-shot SVM](#voc-svm--low-shot-svm)
    - [线性评估](#%E7%BA%BF%E6%80%A7%E8%AF%84%E4%BC%B0)
    - [ImageNet半监督分类](#imagenet%E5%8D%8A%E7%9B%91%E7%9D%A3%E5%88%86%E7%B1%BB)
    - [ImageNet最邻近分类](#imagenet%E6%9C%80%E9%82%BB%E8%BF%91%E5%88%86%E7%B1%BB)
  - [检测](#%E6%A3%80%E6%B5%8B)
  - [分割](#%E5%88%86%E5%89%B2)

首先，你应该通过`tools/model_converters/extract_backbone_weights.py`提取你的 backbone 权重。

```shell
python ./tools/model_converters/extract_backbone_weights.py {CHECKPOINT} {MODEL_FILE}
```

参数：

- `CHECKPOINT`：selfsup 方法的权重文件，名称为 epoch\_\*.pth 。
- `MODEL_FILE`：输出的 backbone 权重文件。如果没有指定，下面的 `PRETRAIN` 会使用这个提取的模型文件。

## 分类

关于分类，我们在`tools/benchmarks/classification/`文件夹中提供了脚本，其中有 4 个 `.sh` 文件，1 个用于 VOC SVM 相关的分类任务的文件夹，1 个用于 ImageNet 最邻近分类任务的文件夹。

### VOC SVM / Low-shot SVM

为了运行这个基准评测，你应该首先准备你的 VOC 数据集，数据准备的细节请参考[prepare_data.md](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/prepare_data.md)。

为了评估预训练的模型，你可以运行以下命令。

```shell
# 分布式版本
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm 版本
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

此外，如果你想评估 runner 保存的 ckpt 文件，你可以运行下面的命令。

```shell
# 分布式版本
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm 版本
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**用ckpt测试时，代码使用epoch\_\*.pth文件，不需要提取权重。**

备注：

- `${SELFSUP_CONFIG}`是自监督实验的配置文件。
- `${FEATURE_LIST}`是一个字符串，指定 layer1 到 layer5 的特征用于评估；例如，如果你只想评估 layer5 ，那么 `FEATURE_LIST` 是 "feat5"，如果你想评估所有的特征，那么`FEATURE_LIST`是 "feat1 feat2 feat3 feat4 feat5"（用空格分隔）。如果留空，默认`FEATURE_LIST`为 "feat5"。
- `PRETRAIN`：预训练的模型文件。
- 如果你想改变GPU的数量，你可以在命令的开头加上`GPUS_PER_NODE=4 GPUS=4`。
- `EPOCH`是你要测试的 ckpt 的 epoch 数。

### 线性评估

线性评估是最通用的基准评测之一，我们整合了几篇论文的配置设置，也包括多头线性评估。我们在自己的代码库中为多头功能编写分类模型，因此，为了运行线性评估，我们仍然使用 `.sh` 脚本来启动训练。支持的数据集是**ImageNet**、**Places205**和**iNaturalist18**。

```shell
# 分布式版本
bash tools/benchmarks/classification/dist_train_linear.sh ${CONFIG} ${PRETRAIN}

# slurm 版本
bash tools/benchmarks/classification/slurm_train_linear.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

备注：

- 默认的 GPU 数量是 8，当改变 GPUS 时，也请相应改变配置文件中的 `samples_per_gpu` ，以确保总 batch size 为256。
- `CONFIG`: 使用 `configs/benchmarks/classification/` 下的配置文件。具体有`imagenet`（不包括 `imagenet_*percent` 文件夹）， `places205` 和`inaturalist2018`。
- `PRETRAIN`：预训练的模型文件。

### ImageNet半监督分类

为了运行 ImageNet 半监督分类，我们仍然使用 `.sh` 脚本来启动训练。

```shell
# 分布式版本
bash tools/benchmarks/classification/dist_train_semi.sh ${CONFIG} ${PRETRAIN}

# slurm 版本
bash tools/benchmarks/classification/slurm_train_semi.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

备注：

- 默认的 GPU 数量是4。
- `CONFIG`: 使用 `configs/benchmarks/classification/imagenet/` 下的配置文件，名为 `imagenet_*percent` 文件夹。
- `PRETRAIN`：预训练的模型文件。

### ImageNet最邻近分类

为了使用最邻近基准评测来评估预训练的模型，你可以运行以下命令。

```shell
# 分布式版本
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn_pretrain.sh ${SELFSUP_CONFIG} ${PRETRAIN}

# slurm 版本
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN}
```

此外，如果你想评估 runner 保存的 ckpt 文件，你可以运行下面的命令。

```shell
# 分布式版本
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn_epoch.sh ${SELFSUP_CONFIG} ${EPOCH}

# slurm 版本
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH}
```

**用ckpt测试时，代码使用epoch\_\*.pth文件，不需要提取权重。**

备注：

- `${SELFSUP_CONFIG}`是自监督实验的配置文件。
- `PRETRAIN`：预训练的模型文件。
- 如果你想改变GPU的数量，你可以在命令的开头加上`GPUS_PER_NODE=4 GPUS=4`。
- `EPOCH`是你要测试的 ckpt 的 epoch 数。

## 检测

在这里，我们倾向于使用 MMDetection 来完成检测任务。首先，确保你已经安装了[MIM](https://github.com/open-mmlab/mim)，它也是OpenMMLab的一个项目。

```shell
pip install openmim
```

安装该软件包非常容易。

此外，请参考MMDet的[安装](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)和[数据准备](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md)

安装完成后，你可以用简单的命令运行 MMDet

```shell
# 分布式版本
bash tools/benchmarks/mmdetection/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm 版本
bash tools/benchmarks/mmdetection/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

备注：

- `CONFIG`：使用 `configs/benchmarks/mmdetection/` 下的配置文件或编写你自己的配置文件。
- `PRETRAIN`: 预训练的模型文件。

或者如果你想用[detectron2](https://github.com/facebookresearch/detectron2)做检测任务，我们也提供一些配置文件。
请参考[INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)进行安装，并按照[目录结构](https://github.com/facebookresearch/detectron2/tree/main/datasets)来准备 detectron2 所需的数据集。

````shell
conda activate detectron2 # 在这里使用 detectron2 环境，否则使用 open-mmlab 环境
cd benchmarks/detection
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # 必须使用 .pkl 作为输出文件扩展名
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```

## 分割

对于语义分割任务，我们使用的是 MMSegmentation 。首先，确保你已经安装了[MIM](https://github.com/open-mmlab/mim)，它也是OpenMMLab的一个项目。

```shell
pip install openmim
```
安装该软件包非常容易。

此外，请参考 MMSeg 的[安装](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md)和[数据准备](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets)。

安装后，你可以用简单的命令运行 MMSeg

```shell
#分布式版本
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm 版本
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

备注：
- `CONFIG`：使用 `configs/benchmarks/mmsegmentation/` 下的配置文件或编写自己的配置文件。
- `PRETRAIN`：预训练的模型文件。
````
