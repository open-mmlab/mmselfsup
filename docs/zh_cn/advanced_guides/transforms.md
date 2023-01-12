# 数据变化

- [数据变化](#数据变化)
  - [数据变换概述](#数据变换概述)
  - [MultiView 简介](#multiview-简介)
  - [PackSelfSupInputs 简介](#packselfsupinputs-简介)

## 数据变换概述

在 [add_transforms](./add_transforms.md) 中我们介绍了如何构建 `Pipeline` 。 `Pipeline` 里有一系列的数据变换。MMSelfSup 中数据变换主要分为三类：

1. 处理数据用到的数据变换。[processing.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/processing.py) 中定义了独特的数据变换，比如`RandomCrop`, `RandomResizedCrop` 和 `RandomGaussianBlur`。我们也可以用其它仓库的数据变换，比如 MMCV 中的 `LoadImageFromFile`。

2. 不同视角看同一照片的数据变换打包器。这个定义在 [wrappers.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/wrappers.py)。

3. 将数据变换使得数据能输入算法中。这个定义在 [formatting.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/formatting.py)。

总的来说，我们用的是如下的这些数据变换。我们将详细讨论最后两种数据变换。

|                                                      类别                                                       |                            作用                            |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------: |
|                           [`BEiTMaskGenerator`](mmselfsup.datasets.BEiTMaskGenerator)                           |             为图像产生随机掩码，参考自 `BEiT`              |
|                         [`SimMIMMaskGenerator`](mmselfsup.datasets.SimMIMMaskGenerator)                         |             产生随机块状掩码，参考自 `SimMIM`              |
|                                 [`ColorJitter`](mmselfsup.datasets.ColorJitter)                                 |           随机改变图像亮度，对比度，饱和度和色调           |
|                                  [`RandomCrop`](mmselfsup.datasets.RandomCrop)                                  |                        随机裁切图像                        |
|                          [`RandomGaussianBlur`](mmselfsup.datasets.RandomGaussianBlur)                          |              随机高斯模糊，参考自， `SimCLR`               |
|                           [`RandomResizedCrop`](mmselfsup.datasets.RandomResizedCrop)                           |             随机裁切图像，并调整大小到特定比例             |
| [`RandomResizedCropAndInterpolationWithTwoPic`](mmselfsup.datasets.RandomResizedCropAndInterpolationWithTwoPic) | 随机裁切图像，并调整大小到特定比例，可以给定不同的插值方法 |
|                              [`RandomSolarize`](mmselfsup.datasets.RandomSolarize)                              |                随机曝光调整，参考自 `BYOL`                 |
|                          [`RotationWithLabels`](mmselfsup.datasets.RotationWithLabels)                          |                          旋转预测                          |
|                       [`RandomPatchWithLabels`](mmselfsup.datasets.RandomPatchWithLabels)                       |                          随机分块                          |
|                              [`RandomRotation`](mmselfsup.datasets.RandomRotation)                              |                        随机旋转图像                        |
|                             [`MultiView`](mmselfsup.datasets.transforms.MultiView)                              |                     多角度图像的封装器                     |
|                           [`PackSelfSupInputs`](mmselfsup.datasets.PackSelfSupInputs)                           |                打包数据为可以送入算法的格式                |

## MultiView 简介

我们为一些算法定义了名为 [`MultiView`](mmselfsup.datasets.transforms.MultiView) 的多角度照片输入的封装器，比如 MoCo 系列，SimCLR，SwAV 等。在配置文件中，我们能这样定义：

```python
pipeline = [
     dict(type='MultiView',
          num_views=2,
          transforms=[
            [dict(type='Resize', scale=224),]
          ])
]
```

这意味着数据管道里面有两个角度。

我们也可以这样定义有不同角度的数据管道：

```python
pipeline = [
     dict(type='MultiView',
          num_views=[2, 6],
          transforms=[
            [
              dict(type='Resize', scale=224)],
            [
              dict(type='Resize', scale=224),
              dict(type='RandomSolarize')],
          ])
]
```

这意味着有两个数据管道，他们分别有两个角度和六个角度。在 [imagenet_mocov1.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_mocov1.py) 和 [imagenet_mocov2.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_mocov2.py) 和 [imagenet_swav_mcrop-2-6.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_swav_mcrop-2-6.py) 中有更多例子。

## PackSelfSupInputs 简介

我们定义了一个名为 [`PackSelfSupInputs`](mmselfsup.datasets.transforms.PackSelfSupInputs) 的类来将数据转换为能输入算法中的格式。这种转换通常在数据管道的最后，就像下面这样：

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```
