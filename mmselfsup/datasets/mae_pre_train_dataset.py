# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource


@DATASETS.register_module()
class MAEDataset(BaseDataset):
    """The dataset outputs the augmented image for MAE pre-training.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[[dict]]): A list of data augmentations,
            where each augmentaion contains element that represents
            an operation defined in `mmselfsup.datasets.pipelines.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    """

    def __init__(self, data_source, pipeline, prefetch=False):
        self.data_source = build_datasource(data_source)
        self.trans = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
        self.prefetch = prefetch

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        img = self.trans(img)

        return dict(img=img)

    def evaluate(self, results, logger=None):
        return NotImplemented
