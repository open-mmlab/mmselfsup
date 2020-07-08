import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .registry import DATASETS
from .base import BaseDataset


def image_to_patches(img):
    split_per_side = 3
    patch_jitter = 21
    h, w = img.size
    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter

    patches = []

    for i in range(split_per_side):
        for j in range(split_per_side):
            p = TF.crop(img, i * h_grid, j * w_grid, h_grid, w_grid)
            if h_patch < h_grid or w_patch < w_grid:
                p = transforms.RandomCrop((h_patch, w_patch))(p)
            p = TF.to_tensor(p)
            p = TF.normalize(p,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            patches.append(p)

    return patches


@DATASETS.register_module
class RelativeLocDataset(BaseDataset):
    """Dataset for relative patch location
    """

    def __init__(self, data_source, pipeline):
        super(RelativeLocDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img = self.pipeline(img)
        patches = image_to_patches(img)
        perms = []
        [perms.append(torch.cat((patches[i], patches[4]), dim=0)) for i in range(9) if i != 4]
        patch_labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        return dict(img=torch.stack(perms), patch_label=patch_labels)  # 8(2C)HW, 8

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
