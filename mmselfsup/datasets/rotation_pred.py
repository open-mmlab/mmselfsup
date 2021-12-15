# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import BaseDataset
from .builder import DATASETS
from .utils import to_numpy


def rotate(img):
    """Rotate input image with 0, 90, 180, and 270 degrees.

    Args:
        img (Tensor): input image of shape (C, H, W).

    Returns:
        list[Tensor]: A list of four rotated images.
    """
    return [
        img,
        torch.flip(img.transpose(1, 2), [1]),
        torch.flip(img, [1, 2]),
        torch.flip(img, [1]).transpose(1, 2)
    ]


@DATASETS.register_module()
class RotationPredDataset(BaseDataset):
    """Dataset for rotation prediction.

    The dataset rotates the image with 0, 90, 180, and 270 degrees and outputs
    labels `0, 1, 2, 3` correspodingly.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(RotationPredDataset, self).__init__(data_source, pipeline,
                                                  prefetch)

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        img = torch.stack(rotate(img), dim=0)
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return dict(img=img, rot_label=rotation_labels)

    def evaluate(self, results, logger=None):
        return NotImplemented
