# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import BaseDataset
from .builder import DATASETS
from .utils import to_numpy


@DATASETS.register_module()
class DeepClusterDataset(BaseDataset):
    """Dataset for DC and ODC.

    The dataset initializes clustering labels and assigns it during training.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(DeepClusterDataset, self).__init__(data_source, pipeline,
                                                 prefetch)
        # init clustering labels
        self.clustering_labels = [-1 for _ in range(len(self.data_source))]

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        img = self.pipeline(img)
        clustering_label = self.clustering_labels[idx]
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, pseudo_label=clustering_label, idx=idx)

    def assign_labels(self, labels):
        assert len(self.clustering_labels) == len(labels), (
            f'Inconsistent length of assigned labels, '
            f'{len(self.clustering_labels)} vs {len(labels)}')
        self.clustering_labels = labels[:]

    def evaluate(self, results, logger=None):
        return NotImplemented
