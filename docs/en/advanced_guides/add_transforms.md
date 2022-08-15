# Tutorial 2: Customize Data Pipelines

- [Tutorial 2: Customize Data Pipelines](#tutorial-2-customize-data-pipelines)
  - [Overview of `Pipeline`](#overview-of-pipeline)
  - [Creating new augmentations in `Pipeline`](#creating-new-augmentations-in-pipeline)

## Overview of `Pipeline`

`DataSource` and `Pipeline` are two important components in `Dataset`. We have introduced `DataSource` in [add_new_dataset](./1_new_dataset.md). And the `Pipeline` is responsible for applying a series of data augmentations to images, such as random flip.

Here is a config example of `Pipeline` for `SimCLR` training:

```python
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5)
]
```

Every augmentation in the `Pipeline` receives an image as input and outputs an augmented image.

## Creating new augmentations in `Pipeline`

1.Write a new transformation function in [transforms.py](../../mmselfsup/datasets/pipelines/transforms.py) and overwrite the `__call__` function, which takes a `Pillow` image as input:

```python
@PIPELINES.register_module()
class MyTransform(object):

    def __call__(self, img):
        # apply transforms on img
        return img
```

2.Use it in config files. We reuse the config file shown above and add `MyTransform` to it.

```python
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='MyTransform'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5)
]
```
