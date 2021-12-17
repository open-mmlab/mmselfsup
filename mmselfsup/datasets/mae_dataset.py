# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.utils import build_from_cfg
from torch.functional import norm
from torchvision.transforms import Compose
from einops import rearrange

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


class RandomMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


@DATASETS.register_module()
class MAEDataset(BaseDataset):
    """The dataset outputs the augmented image and the mask.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[[dict]]): A list of data augmentations,
            where each augmentaion contains element that represents
            an operation defined in `mmselfsup.datasets.pipelines.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MAEDataset(data_source, [pipeline])
        >>> img, mask = dataset[idx]
        The dataset will return the augmented image and the mask
    """

    def __init__(self,
                 data_source,
                 pipeline,
                 target_pipeline,
                 window_size,
                 mask_ratio,
                 prefetch=False,
                 normalize_target=True):
        self.data_source = build_datasource(data_source)
        self.trans = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
        self.target_trans = Compose([build_from_cfg(p, PIPELINES) for p in target_pipeline])

        self.prefetch = prefetch
        self.masked_position_generator = RandomMaskingGenerator(
            window_size, mask_ratio)
        self.window_size = window_size
        self.normalize_target = normalize_target

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        img = self.trans(img)
        target = self.target_trans(img)
        _, h, _ = img.shape
        patch_size = h // self.window_size

        if self.normalize_target:
            target = rearrange(target, 'c (h p1) (w p2) -> (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
            target = (target - target.mean(dim=-2, keepdim=True)
                    ) / (target.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            target = rearrange(target, 'n p c -> n (p c)')
        else:
            target = rearrange(target, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        mask = torch.from_numpy(self.masked_position_generator()).to(torch.bool)
        _, C = target.shape
        target = target[mask].reshape(-1, C)

        return dict(img=img, mask=mask, target=target)

    def evaluate(self, results, logger=None):
        return NotImplemented
