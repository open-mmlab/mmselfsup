# 教程 2：自定义数据管道

- [教程 2：自定义数据管道](#%E6%95%99%E7%A8%8B-2-%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E7%AE%A1%E9%81%93)
  - [`Pipeline` 概览](#Pipeline-%E6%A6%82%E8%A7%88)
  - [在 `Pipeline` 中创建新的数据增强](#%E5%9C%A8-Pipeline-%E4%B8%AD%E5%88%9B%E5%BB%BA%E6%96%B0%E7%9A%84%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA)

## `Pipeline` 概览

`DataSource` 和 `Pipeline` 是 `Dataset` 的两个重要组件。我们已经在 [add_new_dataset](./1_new_dataset.md) 中介绍了 `DataSource` 。  `Pipeline` 负责对图像进行一系列的数据增强，例如随机翻转。

这是用于 `SimCLR` 训练的 `Pipeline` 的配置示例：

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

`Pipeline` 中的每个增强都接收一张图像作为输入，并输出一张增强后的图像。

## 在 `Pipeline` 中创建新的数据增强

1.在 [transforms.py](../../mmselfsup/datasets/pipelines/transforms.py) 中编写一个新的数据增强函数，并覆盖 `__call__` 函数，该函数接收一张 `Pillow` 图像作为输入：

```python
@PIPELINES.register_module()
class MyTransform(object):

    def __call__(self, img):
        # apply transforms on img
        return img
```

2.在配置文件中使用它。我们重新使用上面的配置文件，并在其中添加 `MyTransform`。

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
