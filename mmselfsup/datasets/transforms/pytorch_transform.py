# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from mmselfsup.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MAERandomResizedCrop(transforms.RandomResizedCrop):
    """RandomResizedCrop for matching TF/TPU implementation: no for-loop is
    used.

    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206 # noqa: E501
    """

    @staticmethod
    def get_params(img: Image.Image, scale: tuple, ratio: tuple) -> Tuple:
        width, height = img.size
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1, )).item()
        j = torch.randint(0, width - w + 1, size=(1, )).item()

        return i, j, h, w

    def forward(self, results: dict) -> dict:
        """The forward function of MAERandomResizedCrop.

        Args:
            results (dict): The results dict contains the image and all these
                information related to the image.

        Returns:
            dict: The results dict contains the cropped image and all these
            information related to the image.
        """
        img = results['img']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        results['img'] = img
        return results
