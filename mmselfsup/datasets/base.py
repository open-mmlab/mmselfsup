# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod

from mmcv.utils import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .builder import PIPELINES, build_datasource


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset class.

    The base dataset can be inherited by different algorithm's datasets. After
    `__init__`, the data source and pipeline will be built. Besides, the
    algorithm specific dataset implements different operations after obtaining
    images from data sources.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        warnings.warn('The dataset part will be refactored, it will soon '
                      'support `dict` in pipelines to save more information, '
                      'the same as the pipeline in `MMDet`.')
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.CLASSES = self.data_source.CLASSES

    def __len__(self):
        return len(self.data_source)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def evaluate(self, results, logger=None, **kwargs):
        pass
