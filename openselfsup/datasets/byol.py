import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource


@DATASETS.register_module
class BYOLDataset(Dataset):
    """Dataset for BYOL.
    """

    def __init__(self, data_source, pipeline1, pipeline2):
        self.data_source = build_datasource(data_source)
        pipeline1 = [build_from_cfg(p, PIPELINES) for p in pipeline1]
        self.pipeline1 = Compose(pipeline1)
        pipeline2 = [build_from_cfg(p, PIPELINES) for p in pipeline2]
        self.pipeline2 = Compose(pipeline2)

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img1 = self.pipeline1(img)
        img2 = self.pipeline2(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
