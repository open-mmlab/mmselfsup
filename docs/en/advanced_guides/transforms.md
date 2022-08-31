# Transforms

- [Transforms]()
  - [Overview of transforms](#overview-of-transforms)
  - [Introduction of `MultiView`](#introduction-of-multiview)
  - [Introduction of `PackSelfSupInputs`](#introduction-of-packselfsupinputs)


## Overview of transforms
We have introduced how to build a `Pipeline` in [add_transforms](./add_transforms.md). A `Pipeline` contains a series of
`transforms`. There are three main categories of `transforms` in MMSelfSup:
1. Transforms about processing the data. The unique transforms in MMSelfSup are defined in [processing.py](../../../mmselfsup/datasets/transforms/processing.py), e.g. `RandomCrop`, `RandomResizedCrop` and `RandomGaussianBlur`.
We may also use some transforms from other repositories, e.g. `LoadImageFromFile` from MMCV.
2. The transform wrapper for multiple views of an image. It is defined in [wrappers.py](../../../mmselfsup/datasets/transforms/wrappers.py).
3. The transform to pack data into the format compatible with the inputs of algorithm. It is defined in [formatting.py](../../../mmselfsup/datasets/transforms/formatting.py).

The last two transforms will be introduced below.

## Introduction of `MultiView`
We build a wrapper named `MultiView` for some algorithms e.g. MOCO, SimCLR and SwAV with multi-view image inputs. In the config file, we can 
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
More examples can be found in [imagenet_mocov1.py](../../../configs/selfsup/_base_/datasets/imagenet_mocov1.py), [imagenet_mocov2.py](../../../configs/selfsup/_base_/datasets/imagenet_mocov2.py) and [imagenet_swav_mcrop-2-6.py](../../../configs/selfsup/_base_/datasets/imagenet_swav_mcrop-2-6.py) etc.  

## Introduction of `PackSelfSupInputs`
We build a class named `PackSelfSupInputs` to pack data into the format compatible with the inputs of algorithm. This transform
is usually put at the end of the pipeline like:
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```