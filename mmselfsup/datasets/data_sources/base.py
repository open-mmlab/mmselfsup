# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
from PIL import Image


class BaseDataSource(object, metaclass=ABCMeta):
    """Datasource base class to load dataset information.

    Args:
        data_prefix (str): the prefix of data path.
        classes (str | Sequence[str], optional): Specify classes to load.
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix.
        test_mode (bool): in train mode or test mode. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to color.
        channel_order (str): The channel order of images when loaded. Defaults
            to rgb.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
    """

    CLASSES = None

    def __init__(self,
                 data_prefix,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args
        self.file_client = None
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.data_infos)

    @abstractmethod
    def load_annotations(self):
        pass

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return self.data_infos[idx]['gt_label'].astype(np.int)

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_img(self, idx):
        """Get image by index.

        Args:
            idx (int): Index of data.

        Returns:
            Image: PIL Image format.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if self.data_infos[idx].get('img_prefix', None) is not None:
            if self.data_infos[idx]['img_prefix'] is not None:
                filename = osp.join(
                    self.data_infos[idx]['img_prefix'],
                    self.data_infos[idx]['img_info']['filename'])
            else:
                filename = self.data_infos[idx]['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order=self.channel_order)
        else:
            img = self.data_infos[idx]['img']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        img = img.astype(np.uint8)
        return Image.fromarray(img)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names
