# Add Datasets

In this tutorial, we introduce the basic steps to create your customized dataset. Before learning to create your customized datasets, it is recommended to learn the basic concept of datasets in file [datasets.md](datasets.md).

- [Add Datasets](#add-datasets)
  - [Step 1: Creating the Dataset](#step-1-creating-the-dataset)
  - [Step 2: Add NewDataset to \_\_init\_\_py](#step-2-add-newdataset-to-__init__py)
  - [Step 3: Modify the config file](#step-3-modify-the-config-file)

If your algorithm does not need any customized dataset, you can use these off-the-shelf datasets under [datasets directory](mmselfsup.datasets). But to use these existing datasets, you have to convert your dataset to existing dataset format.

As for image pretraining, it is recommended to follow the format of MMClassification.

## Step 1: Creating the Dataset

You could implement a new dataset class, inherited from `CustomDataset` from MMClassification for image pretraining.

Assume the name of your `Dataset` is `NewDataset`, you can create a file, named `new_dataset.py` under `mmselfsup/datasets` and implement `NewDataset` in it.

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

## Step 2: Add NewDataset to \_\_init\_\_py

Then, add `NewDataset` in `mmselfsup/dataset/__init__.py`. If it is not imported, the `NewDataset` will not be registered successfully.

```python
...
from .new_dataset import NewDataset

__all__ = [
    ..., 'NewDataset'
]
```

## Step 3: Modify the config file

To use `NewDataset`, you can modify the config as the following:

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
