from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import Dataset

from openselfsup.utils import print_log, build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_source (dict): Data source defined in
            `openselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `oenselfsup.datasets.pipelines`.
    """

    def __init__(self, data_source, pipeline):
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return self.data_source.get_length()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def evaluate(self, scores, keyword, logger=None, **kwargs):
        pass
