# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class LowFreqTargetGenerator(BaseModule):
    """Generate low-frquency target for images.

    This module is used in PixMIM: Rethinking Pixel Reconstruction in Masked
    Image Modeling to remove these high-frequency information from images.

    Args:
        radius (int): radius of low pass filter.
        img_size (Union[int, Tuple[int, int]]): size of input images.
    """

    def __init__(self, radius: int, img_size: Union[int, Tuple[int,
                                                               int]]) -> None:
        super().__init__()
        self.radius = radius
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size,
                                                                      img_size)
        # generate low pass filter
        low_pass_filter = self._generate_low_pass_filter()
        self.register_buffer('low_pass_filter', low_pass_filter)

    def _generate_low_pass_filter(self) -> torch.Tensor:
        """Generate low pass filter.

        This low pass filter is a ideal circular low pass filter. The band
        width (radius) of this filter is in the range of
        [0, \\frac{1}{2}min(h, w)].

        Returns:
            torch.Tensor: low pass filter.
        """
        h, w = self.img_size
        low_pass_filter = torch.ones((3, h, w))
        for i in range(h):
            for j in range(w):
                if (i - (h - 1) / 2)**2 + (j -
                                           (w - 1) / 2)**2 > self.radius**2:
                    low_pass_filter[:, i, j] = 0
        return low_pass_filter

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Filter out these high frequency components from images.

        Args:
            imgs (torch.Tensor): input images, which has shape (N, C, H, W).

        Returns:
            torch.Tensor: low frequency target, which has the same shape as
                input images.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1,
                                                        1).to(imgs.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1,
                                                       1).to(imgs.device)
        # recover the image to the pre-normalized form
        imgs = imgs * std + mean

        freq_imgs = torch.fft.fft2(imgs)
        freq_imgs = torch.fft.fftshift(freq_imgs, dim=(-2, -1))

        # low pass images
        low_pass_imgs = freq_imgs * self.low_pass_filter
        low_pass_imgs = torch.fft.ifft2(low_pass_imgs)
        low_pass_imgs = torch.abs(low_pass_imgs)

        low_pass_imgs = (low_pass_imgs - mean) / std

        return low_pass_imgs
