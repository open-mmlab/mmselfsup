import torch

from .registry import DATASETS
from .base import BaseDataset


def rotate(img):
    '''
    img: Tensor(CHW)
    '''
    return [
        img,
        torch.flip(img.transpose(1, 2), [1]),
        torch.flip(img, [1, 2]),
        torch.flip(img, [1]).transpose(1, 2)
    ]


@DATASETS.register_module
class RotationPredDataset(BaseDataset):
    """Dataset for rotation prediction 
    """

    def __init__(self, data_source, pipeline):
        super(RotationPredDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        img = torch.stack(rotate(img), dim=0)
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return dict(img=img, rot_label=rotation_labels)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
