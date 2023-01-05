# 数据集

- [数据集](#数据集)
  - [数据集](#数据集-1)
    - [重构个人数据集](#重构个人数据集)
    - [在个人 Config 中调用 OpenMMLab 其他代码库的数据集](#在个人-config-中调用-openmmlab-其他代码库的数据集)
  - [采样器](#采样器)
  - [数据变换](#数据变换)

`mmselfsup` 算法库中的 `datasets` 文件夹包罗了各种与数据加载相关的模块文件。
此文件夹主要分为如下三个部分：

- 自定义数据集，用于图像读取与加载
- 自定义数据集采样器，用于图像加载之前进行索引的读取
- 数据变换工具，用于在数据输入模型之前进行数据增强，如 `RandomResizedCrop`

在本教程中，我们将对三个部分依次进行较为详尽的解释。

## 数据集

OpenMMLab 开源算法体系为用户提供了海量开箱即用的数据集，这些数据集都与 [BaseDataset](https://github.com/open-mmlab/mmengine/blob/429bb27972bee1a9f3095a4d5f6ac5c0b88ccf54/mmengine/dataset/base_dataset.py#L116) 一脉相承，
并在 [MMEngine](https://github.com/open-mmlab/mmengine) 中得以实现。 如想进一步了解 `BaseDataset` 中的各项功能，感兴趣的用户可以参考 [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html) 中的文档。对于 `MMSelfSup`， `ImageNet`、`ADE20KDataset`  和 `CocoDataset` 是三个较为常用的数据集。 在起步之前，用户需要对文件夹进行一些前置的重构工作，具体指南如下所述。

### 重构个人数据集

万事俱备，只欠东风。使用准备好的这些数据集，用户需要将数据集重构为如下格式。

```
mmselfsup
├── mmselfsup
├── tools
├── configs
├── docs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   │
│   │── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │
│   │── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

更为详尽的注释文件以及各子文件夹的结构，可以参考 OpenMMLab 的其他代码库，如 [MMClassfication](https://github.com/open-mmlab/mmclassification),
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 和 [MMDetection](https://github.com/open-mmlab/mmdetection).

### 在个人 Config 中调用 OpenMMLab 其他代码库的数据集

```python
# 调用 MMClassification 中的 ImageNet 数据集
# 在数据加载器中调用 ImageNet
# 为简单起见，我们只提供与从 MMClassification 导入 ImageNet 的相关 config
# 而不是数据加载器的全量的 config
# ``mmcls`` 前缀传达给 ``Registry`` 需要在 MMClassification 中搜索 ``ImageNet``
train_dataloader=dict(dataset=dict(type='mmcls.ImageNet', ...), ...)
```

```python
# 调用 MMSegmentation 中的 ADE20KDataset 数据集
# 在数据加载器中使用 ADE20KDataset
# 为简单起见，我们只提供与从 MMSegmentation 导入 ADE20KDataset 的相关 config
# 而不是数据加载器的全量的 config
# ``mmseg`` 前缀传达给 ``Registry`` 需要在 MMSegmentation 中搜索 ``ADE20KDataset``
train_dataloader=dict(dataset=dict(type='mmseg.ADE20KDataset', ...), ...)
```

```python
# 在数据加载器中调用 CocoDataset
# 为简单起见，我们只提供与从 MMDetection 导入 CocoDataset 的相关 config
# 而不是数据加载器的全量的 config
# ``mmdet`` 前缀传达给 ``Registry`` 需要在 MMDetection 中搜索 ``CocoDataset``
train_dataloader=dict(dataset=dict(type='mmdet.CocoDataset', ...), ...)
```

```python
# 在 MMSelfSup 中调用数据集，如 ``DeepClusterImageNet``
train_dataloader=dict(dataset=dict(type='DeepClusterImageNet', ...), ...)
```

通过上文，我们介绍了调用数据集的两个关键的步骤，希望用户可以掌握如何在 MMSelfSup 中使用数据集的相关基本概念。 如果用户有创建自定义数据集的意愿，可参考文档 [add_datasets](./add_datasets.md)。

## 采样器

在 pytorch 中，`Sampler` 用于在加载之前对数据的索引进行采样。 `MMEngine` 中已经实现和开源了 `DefaultSampler` 和
`InfiniteSampler`。 大多数情况下，我们可以直接调用，无需手动去实现自定义采样器。 然而 `DeepClusterSampler` 是一个值得一提的特例，因为其中纳入了进行唯一索引采样的逻辑， 因此，如果用户想对此采样器的相关信息一览无遗，则可进一步参考我们的 API 文档。 如果你有自行实现自定义采样器的更进一步的想法，同样可以参考 `DeepClusterSampler` 在 `samplers` 文件夹进行实现。

## 数据变换

简而言之，`transform` 是指 `MM-repos` 中的数据变换模块，我们将一系列的 transform 组合成了一个列表，即 `pipeline`。
`MMCV` 中已经完善了一些涵盖大多数场景的变换， 此外，每个 `MM-repo` 也都遵循 MMCV 中的 [用户指南](https://github.com/open-mmlab/mmcv/blob/dev-2.x/docs/zh_cn/understand_mmcv/data_transform.md) 定义了自己的变换。 实操而言，每个自定义的数据集需要：i) 继承 [BaseTransform](https://github.com/open-mmlab/mmcv/blob/19a024155a0b710568c2faeae07dead2a5550392/mmcv/transforms/base.py#L6)，
ii) 覆盖 `transform` 函数并在其中实现自行设计的关键逻辑。 在 MMSelfSup 中，我们已经实现了如下这些变换：

|                                                      class                                                      |
| :-------------------------------------------------------------------------------------------------------------: |
|                           [`PackSelfSupInputs`](mmselfsup.datasets.PackSelfSupInputs)                           |
|                           [`BEiTMaskGenerator`](mmselfsup.datasets.BEiTMaskGenerator)                           |
|                         [`SimMIMMaskGenerator`](mmselfsup.datasets.SimMIMMaskGenerator)                         |
|                                 [`ColorJitter`](mmselfsup.datasets.ColorJitter)                                 |
|                                  [`RandomCrop`](mmselfsup.datasets.RandomCrop)                                  |
|                          [`RandomGaussianBlur`](mmselfsup.datasets.RandomGaussianBlur)                          |
|                           [`RandomResizedCrop`](mmselfsup.datasets.RandomResizedCrop)                           |
| [`RandomResizedCropAndInterpolationWithTwoPic`](mmselfsup.datasets.RandomResizedCropAndInterpolationWithTwoPic) |
|                              [`RandomSolarize`](mmselfsup.datasets.RandomSolarize)                              |
|                          [`RotationWithLabels`](mmselfsup.datasets.RotationWithLabels)                          |
|                       [`RandomPatchWithLabels`](mmselfsup.datasets.RandomPatchWithLabels)                       |
|                              [`RandomRotation`](mmselfsup.datasets.RandomRotation)                              |

对于感兴趣的社区用户，可以参考 API 文档以更为全面了解这些转换。目前为止， 我们已经初步介绍了关于转换的基本概念，
若想进一步了解如何在个人的 config 中使用它们或实现自定义转换，
可以参考文档 ： [transforms](./transforms.md) 和 [add_transforms](./add_transforms.md).
