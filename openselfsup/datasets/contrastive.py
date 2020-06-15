import torch

from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    """Dataset for rotation prediction 
    """

    def __init__(self, data_source, pipeline):
        super(ContrastiveDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img, _ = self.data_source.get_sample(idx)
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
