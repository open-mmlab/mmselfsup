# 添加数据集

在本教程中，我们介绍了创建自定义数据集的基本步骤。在学习创建自定义数据集之前，建议学习文件 [datasets.md](datasets.md) 中数据集的基本概念。

- [添加数据集](#添加数据集)
  - [步骤 1: 创建数据集](#步骤-1-创建数据集)
  - [步骤 2: 添加数据集到 \_\_init\_\_py](#步骤-2-添加数据集到-__init__py)
  - [步骤 3: 修改配置文件](#步骤-3-修改配置文件)

如果您的算法不需要任何自定义数据集类，您可以使用 [datasets directory](mmselfsup.datasets) 里的现成的数据集类。但使用这些现有的数据集类，您必须将您的数据集转换成现有数据集类要求的格式。

关于图像预训练，建议遵循 MMClassification 的格式。

## 步骤 1: 创建数据集

您可以实现一个新的数据集类，它继承自 MMClassification 的 `CustomDataset`，用于图像预训练。

假如您的数据集类为 `NewDataset`，您可以在 `mmselfsup/datasets` 下创建文件 `new_dataset.py` 并在其中实现自定义数据集类 `NewDataset`。

```python
from typing import List, Optional, Union

from mmcls.datasets import CustomDataset

from mmselfsup.registry import DATASETS


@DATASETS.register_module()
class NewDataset(CustomDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 **kwargs) -> None:
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        # Rewrite load_data_list() to satisfy your specific requirement.
        # The returned data_list could include any information you need from
        # data or transforms.

        # writing your code here
        return data_list

```

## 步骤 2: 添加数据集到 \_\_init\_\_py

然后将 `NewDataset` 导入到 `mmselfsup/dataset/__init__.py` 中。如果没有导入，则 `NewDataset` 没有注册 (register) 成功。

```python
...
from .new_dataset import NewDataset

__all__ = [
    ..., 'NewDataset'
]
```

## 步骤 3: 修改配置文件

使用 `NewDataset` 时，您可以参考下面修改配置文件：

```python
train_dataloader = dict(
    ...
    dataset=dict(
        type='NewDataset',
        data_root=your_data_root,
        ann_file=your_data_root,
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
```
