# Add Transforms

- [Add Transforms](#add-transforms)
  - [Overview of Pipeline](#overview-of-pipeline)
  - [Creating new transforms in Pipeline](#creating-new-transforms-in-pipeline)

## Overview of Pipeline

`Pipeline` is an important component in `Dataset`, which is responsible for applying a series of data augmentations to images, such as `RandomResizedCrop`, `RandomFlip`.

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

Every augmentation in the `Pipeline` receives a `Dict` as input and outputs a `Dict` containing the augmented image and other related information.

## Creating new transforms in Pipeline

1. Write a new transform in [transforms.py](https://github.com/open-mmlab/mmselfsup/tree/1.x/mmselfsup/datasets/transforms) and overwrite the `transform` function, which takes a `Dict` as input:

```python
@TRANSFORMS.register_module()
class MyTransform(BaseTransform):
    """Docstring for transform.
    """

    def transform(self, results: Dict) -> Dict:
        # apply transform
        return results
```

Then, add the transform to [__init__.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/mmselfsup/datasets/transforms/__init__.py).

**Note:** For the implementation of transforms, you could apply functions from [mmcv](https://github.com/open-mmlab/mmcv/tree/2.x/mmcv/image).

2. Add `MyTransform` to config.

```python
view_pipeline = [
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    # add `MyTransform`
    dict(type='MyTransform'),
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
        channel_weights=(0.114, 0.587, 0.299)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```
