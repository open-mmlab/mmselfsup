import torch
from PIL import Image

from .registry import DATASETS
from .base import BaseDataset


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


@DATASETS.register_module
class RotationPredDataset(BaseDataset):
    """Dataset for rotation prediction.
    """

    def __init__(self, data_source, pipeline):
        super(RotationPredDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img = self.pipeline(img)
        img = torch.stack(rotate(img), dim=0)
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return dict(img=img, rot_label=rotation_labels)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
