# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Union

from mmcls.datasets import ImageNet

from mmselfsup.registry import DATASETS


@DATASETS.register_module()
class DeepClusterImageNet(ImageNet):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset inherit ImageNet dataset from MMClassification as the
    DeepCluster and Online Deep Clustering algorithm need to initialize
    clustering labels and assign them during training.

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
        # init clustering labels
        self.clustering_labels = [-1 for _ in range(len(self))]

    def assign_labels(self, labels: list) -> None:
        """Assign new labels to `self.clustering_labels`.

        Args:
            labels (list): The new labels.

        Returns:
            None
        """
        assert len(self.clustering_labels) == len(labels), (
            f'Inconsistent length of assigned labels, '
            f'{len(self.clustering_labels)} vs {len(labels)}')
        self.clustering_labels = labels[:]

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        data_info['clustering_label'] = int(self.clustering_labels[idx])
        return self.pipeline(data_info)
