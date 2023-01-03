# 转换

- [转换](<>)
  - [转换概述](#转换概述)
  - [MultiView概论](#MultiView概论)
  - [PackSelfSupInputs概论](#PackSelfSupInputs概论)

## 转换概述

在[add_transforms](./add_transforms.md)中我们介绍了如何建`Pipeline`。`Pipeline`包括一系列的转换。MMSelfSup中转换主要分为三类：

1. 处理数据用到的转换。[processing.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/processing.py)中定义了独特的转换，比如`RandomCrop`, `RandomResizedCrop` and `RandomGaussianBlur`。我们也可以用其它仓库的转换，比如MMCV中的`LoadImageFromFile`。

2.不同视角看同一照片的转换打包器。这个在[wrappers.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/wrappers.py)中定义过。

3.将数据转换使得数据能输入算法中。这个在[formatting.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/formatting.py)中定义过。

总的来说，我们用的是如下的这些转换。我们将详细讨论最后两种转换。

|                                                      类别                                                      |                                      作用                                      |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
|                           [`BEiTMaskGenerator`](mmselfsup.datasets.BEiTMaskGenerator)                           |                      Generate mask for image refers to `BEiT`                      |
|                         [`SimMIMMaskGenerator`](mmselfsup.datasets.SimMIMMaskGenerator)                         |            Generate random block mask for each Image refers to `SimMIM`            |
|                                 [`ColorJitter`](mmselfsup.datasets.ColorJitter)                                 |      Randomly change the brightness, contrast, saturation and hue of an image      |
|                                  [`RandomCrop`](mmselfsup.datasets.RandomCrop)                                  |                     Crop the given Image at a random location                      |
|                          [`RandomGaussianBlur`](mmselfsup.datasets.RandomGaussianBlur)                          |                    GaussianBlur augmentation refers to `SimCLR`                    |
|                           [`RandomResizedCrop`](mmselfsup.datasets.RandomResizedCrop)                           |                Crop the given image to random size and aspectratio                 |
| [`RandomResizedCropAndInterpolationWithTwoPic`](mmselfsup.datasets.RandomResizedCropAndInterpolationWithTwoPic) | Crop the given PIL Image to random size and aspect ratio with random interpolation |
|                              [`RandomSolarize`](mmselfsup.datasets.RandomSolarize)                              |                     Solarization augmentation refers to `BYOL`                     |
|                          [`RotationWithLabels`](mmselfsup.datasets.RotationWithLabels)                          |                                Rotation prediction                                 |
|                       [`RandomPatchWithLabels`](mmselfsup.datasets.RandomPatchWithLabels)                       |                 Apply random patch augmentation to the given image                 |
|                              [`RandomRotation`](mmselfsup.datasets.RandomRotation)                              |                             Rotate the image by angle                              |
|                             [`MultiView`](mmselfsup.datasets.transforms.MultiView)                              |               A wrapper for algorithms with multi-view image inputs                |
|                           [`PackSelfSupInputs`](mmselfsup.datasets.PackSelfSupInputs)                           |         Pack data into a format compatible with the inputs of an algorithm         |

## MultiView概论

我们为一些算法建名为[`MultiView`](mmselfsup.datasets.transforms.MultiView)的多角度照片输入的包装器，比如MOCO,SimCLR和SwAV。在配置文件中，我们能这样定义：

```python
pipeline = [
     dict(type='MultiView',
          num_views=2,
          transforms=[
            [dict(type='Resize', scale=224),]
          ])
]
```

，这意味着通信流水线(pipeline)里面有两个角度。

我们也可以这样定义有不同角度的通信流水线：

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

这意味着有两个通信流水线，他们分别有两个角度和六个角度。在[imagenet_mocov1.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_mocov1.py)和 [imagenet_mocov2.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_mocov2.py) 和[imagenet_swav_mcrop-2-6.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_swav_mcrop-2-6.py)中有更多例子。

## PackSelfSupInputs概论

我们建一个名为[`PackSelfSupInputs`](mmselfsup.datasets.transforms.PackSelfSupInputs)的类来将数据转换为能输入算法中的格式。这种转换通常在通信流水线的后面，就像这样：

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```
