# Add Transforms

In this tutorial, we introduce the basic steps to create your customized transforms. Before learning to create your customized transforms, it is recommended to learn the basic concept of transforms in file [transforms.md](transforms.md).

- [Add Transforms](#add-transforms)
  - [Overview of Pipeline](#overview-of-pipeline)
  - [Creating a new transform in Pipeline](#creating-a-new-transform-in-pipeline)
    - [Step 1: Creating the transform](#step-1-creating-the-transform)
    - [Step 2: Add NewTransform to \_\_init\_\_py](#step-2-add-newtransform-to-__init__py)
    - [Step 3: Modify the config file](#step-3-modify-the-config-file)

## Overview of Pipeline

`Pipeline` is an important component in `Dataset`, which is responsible for applying a series of data augmentations to images, such as `RandomResizedCrop`, `RandomFlip`, etc.

Here is a config example of `Pipeline` for `SimCLR` training:

```python
view_pipeline = [
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```

Every augmentation in the `Pipeline` receives a `dict` as input and outputs a `dict` containing the augmented image and other related information.

## Creating a new transform in Pipeline

Here are the steps to create a new transform.

### Step 1: Creating the transform

Write a new transform in [processing.py](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/mmselfsup/datasets/transforms/processing.py) and overwrite the `transform` function, which takes a `dict` as input:

```python
@TRANSFORMS.register_module()
class NewTransform(BaseTransform):
    """Docstring for transform.
    """

    def transform(self, results: dict) -> dict:
        # apply transform
        return results
```

**Note:** For the implementation of transforms, you could apply functions in [mmcv](https://github.com/open-mmlab/mmcv/tree/2.x/mmcv/image).

### Step 2: Add NewTransform to \_\_init\_\_py

Then, add the transform to [\_\_init\_\_.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/datasets/transforms/__init__.py).

```python
...
from .processing import NewTransform, ...

__all__ = [
    ..., 'NewTransform'
]
```

### Step 3: Modify the config file

To use `NewTransform`, you can modify the config as the following:

```python
view_pipeline = [
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    # add `NewTransform`
    dict(type='NewTransform'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```
