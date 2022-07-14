# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
from mmcls.datasets import CustomDataset
from mmengine import FileClient

from mmselfsup.registry import DATASETS


@DATASETS.register_module()
class ImageList(CustomDataset):
    """The dataset implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.

    An annotation file should be provided, and each line indicates a sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       1. If data is labeled, the annotation file (the first column is the image
       path and the second column is the index of category): ::

            folder_1/xxx.png 0
            folder_1/xxy.png 1
            folder_2/123.png 5
            folder_2/nsdf3.png 3
            ...

        2. If data is unlabeled, the annotation file is: ::

            folder_1/xxx.png
            folder_1/xxy.png
            folder_2/123.png
            folder_2/nsdf3.png
            ...

    Args:
        ann_file (str): Annotation file path. Defaults to None.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (str | dict): Prefix for training data. Defaults
            to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """  # noqa: E501

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
        """Rewrite load_data_list() function for supporting a list of
        annotation files and unlabeled data.

        Returns:
            List[dict]: A list of data information.
        """
        if self.img_prefix is not None:
            file_client = FileClient.infer_client(uri=self.img_prefix)

        assert self.ann_file is not None
        if not isinstance(self.ann_file, list):
            self.ann_file = [self.ann_file]

        data_list = []
        for ann_file in self.ann_file:
            with open(ann_file, 'r') as f:
                self.samples = f.readlines()
            self.has_labels = len(self.samples[0].split()) == 2

            for sample in self.samples:
                info = {'img_prefix': self.img_prefix}
                sample = sample.split()
                info['img_path'] = file_client.join_path(
                    self.img_prefix, sample[0])
                info['img_info'] = {'filename': sample[0]}
                labels = sample[1] if self.has_labels else -1
                info['gt_label'] = np.array(labels, dtype=np.int64)
                data_list.append(info)
        return data_list
