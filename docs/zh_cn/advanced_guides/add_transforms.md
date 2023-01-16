# 添加数据变换

在本教程中, 我们将介绍创建自定义转换的基本步骤。在学习创建自定义转换之前, 建议先了解文件 [transforms.md](transforms.md) 中转换的基本概念。

- [添加数据变换](#添加数据变换)
  - [管道概述](#管道概述)
  - [在管道中创建新转换](#在管道中创建新转换)
    - [步骤 1: 创建转换](#步骤-1-创建转换)
    - [步骤 2: 将新转换添加到 \_\_init\_\_py](#步骤-2-将新转换添加到-__init__py)
    - [步骤 3: 修改配置文件](#步骤-3-修改配置文件)

## 管道概述

在 `Dataset` 中, `Pipeline` 是中的一个重要组件, 主要负责对图像应用一系列数据增强, 例如: `RandomResizedCrop`, `RandomFlip` 等操作。

以下代码是 `Pipeline` 用于 `SimCLR` 训练的配置示例:

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

在这个 `Pipeline` 中, 每个数据增强接收一个 `dict` , 它们作为输入和输出时刻, 包含图像增强以及其他相关信息的 `dict` 。

## 在管道中创建新转换

以下是创建新转换的步骤。

### 步骤 1: 创建转换

在 [processing.py](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/mmselfsup/datasets/transforms/processing.py) 中编写一个新的转换类, 并在类中覆盖这个 `transform` 函数, 这个函数接收一个 `dict` 的对象, 并返回一个 `dict` 对象

```python
@TRANSFORMS.register_module()
class NewTransform(BaseTransform):
    """Docstring for transform.
    """

    def transform(self, results: dict) -> dict:
        # apply transform
        return results
```

**注意**: 对于这些转换的实现, 您可以应用 [mmcv](https://github.com/open-mmlab/mmcv/tree/2.x/mmcv/image) 中的函数。

### 步骤 2: 将新转换添加到 \_\_init\_\_py

然后, 将转换添加到 [\_\_init\_\_.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/datasets/transforms/__init__.py) 。

```python
...
from .processing import NewTransform, ...

__all__ = [
    ..., 'NewTransform'
]
```

### 步骤 3: 修改配置文件

要使用新添加的 `NewTransform`, 你可以按以下的方式修改配置文件:

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
