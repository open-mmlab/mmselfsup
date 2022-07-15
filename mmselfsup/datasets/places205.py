# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmcls.datasets import CustomDataset

from mmselfsup.registry import DATASETS


@DATASETS.register_module()
class Places205(CustomDataset):
    """`Places205 <http://places.csail.mit.edu/downloadData.html>`_ Dataset.

    The dataset supports two kinds of annotation format. More details can be
    found in :class:`CustomDataset`.

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
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)
