# Transforms

- [Transforms](<>)
  - [Overview of transforms](#overview-of-transforms)
  - [Introduction of `MultiView`](#introduction-of-multiview)
  - [Introduction of `PackSelfSupInputs`](#introduction-of-packselfsupinputs)

## Overview of transforms

We have introduced how to build a `Pipeline` in [add_transforms](./add_transforms.md). A `Pipeline` contains a series of
`transforms`. There are three main categories of `transforms` in MMSelfSup:

1. Transforms about processing the data. The unique transforms in MMSelfSup are defined in [processing.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/processing.py), e.g. `RandomCrop`, `RandomResizedCrop` and `RandomGaussianBlur`.
   We may also use some transforms from other repositories, e.g. `LoadImageFromFile` from MMCV.

2. The transform wrapper for multiple views of an image. It is defined in [wrappers.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/wrappers.py).

3. The transform to pack data into a format compatible with the inputs of the algorithm. It is defined in [formatting.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/formatting.py).

In summary, we implement these `transforms` below. The last two transforms will be introduced in detail.

|                                                      class                                                      |                                      function                                      |
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

## Introduction of `MultiView`

We build a wrapper named [`MultiView`](mmselfsup.datasets.transforms.MultiView) for some algorithms e.g. MOCO, SimCLR and SwAV with multi-view image inputs. In the config file, we can
define it as:

```python
pipeline = [
     dict(type='MultiView',
          num_views=2,
          transforms=[
            [dict(type='Resize', scale=224),]
          ])
]
```

, which means that there are two views in the pipeline.

We can also define pipeline with different views like:

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

This means that there are two pipelines, which contain 2 views and 6 views, respectively.
More examples can be found in [imagenet_mocov1.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_mocov1.py), [imagenet_mocov2.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_mocov2.py) and [imagenet_swav_mcrop-2-6.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/_base_/datasets/imagenet_swav_mcrop-2-6.py) etc.

## Introduction of `PackSelfSupInputs`

We build a class named [`PackSelfSupInputs`](mmselfsup.datasets.transforms.PackSelfSupInputs) to pack data into a format compatible with the inputs of an algorithm. This transform
is usually put at the end of the pipeline like:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```
