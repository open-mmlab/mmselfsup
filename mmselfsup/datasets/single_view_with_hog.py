# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from einops import rearrange
from mmcv.utils import build_from_cfg, print_log
from skimage.feature import hog
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES
from .utils import to_numpy


class HogGenerator:
    """HogGenerator get hog features based on `"Histograms of Oriented
    Gradients for Human Detection".

    <https://ieeexplore.ieee.org/abstract/document/1467360/>`_.

    This code is borrowed from
    <https://github.com/mx-mark/VideoTransformer-pytorch/blob/main/mask_generator.py>

    Args:
        orientations (int): Number of orientation bins. Defaults to 9.
        pixels_per_cell (tuple[int, int]): Number of pixels in each cell.
            Defaults to (8, 8).
        cells_per_block (tuple[int, int]): Number of cells in each block.
            Defaults to (1, 1).
        block_norm (str): Block normalization method. Defaults to 'L2'.
        patch_size (int): Size of patch in vit backbone. Defaults to 16.
    """

    def __init__(self, orientations, pixels_per_cell, cells_per_block,
                 block_norm, patch_size):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.patch_size = patch_size

    def _rgb_to_gray(self, rgb):
        """Convert color images to grayscale.

        Args:
            rgb (numpy.ndarray): Input rgb image of shape (H, W, C).
        Returns:
            numpy.ndarray: Gray image of shape (H, W).
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _extract_gray_hog_features(self, image):
        """Function to extract hog feature of gray image.

        Args:
            image (numpy.ndarray): Input gray image of shape (H, W).
        Returns:
            numpy.ndarray: Hog feaetures of shape (pc, ph, pw),
        """
        hog_feature = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=False)
        hog_feature = self._reshape_hog_features(
            hog_feature, image.shape[0] // self.patch_size,
            image.shape[1] // self.patch_size)
        return hog_feature

    def _extract_RGB_hog_features(self, image):
        """Function to extract hog features for RGB images.

        Args:
            image (numpy.ndarray): Input image of shape (height, width, 3).
        Returns:
            numpy.ndarray: Hog feaetures of shape (pc*3, ph, pw),
        """
        hog_features_r = self._extract_gray_hog_features(image[:, :, 0])
        hog_features_g = self._extract_gray_hog_features(image[:, :, 1])
        hog_features_b = self._extract_gray_hog_features(image[:, :, 2])

        hog_features = np.concatenate(
            [hog_features_r, hog_features_g, hog_features_b], axis=0)
        return hog_features

    def _reshape_hog_features(self, hog_features, ph, pw):
        """Function to reshape hog features from (ph*pw*pc) to (pc, ph, pw)."""
        hog_features = rearrange(
            hog_features,
            '(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)',
            ph=ph,
            pw=pw)
        return hog_features.transpose(2, 0, 1)

    def __call__(self, img):
        """Get hog feature of image.

        Args:
            img (numpy.ndarray): Input image of shape (C, H, W)
        Returns:
            numpy.ndarray:
                Hog feaetures of shape (pc, ph, pw) in gray image and
                (pc*3, ph, pw) in rgb image.
                    ph = H // patch_size
                    pw = W // patch_size
                    pc = orientations x (patch_size / pixels_per_cell[0])
                            x (patch_size / pixels_per_cell[1]
        """
        img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            return self._extract_gray_hog_features(self._rgb_to_gray(img))
        elif img.shape[-1] == 3:
            return self._extract_RGB_hog_features(img)
        else:
            raise Exception('Image must be gray or rgb in HogGenerator')


@DATASETS.register_module()
class SingleViewWithHogDataset(BaseDataset):
    """The dataset outputs one view of an image and hog features, containing
    some other information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline_before_hog (list[dict]): A list of dict, where each element
            represents an operation defined in `mmselfsup.datasets.pipelines`.
        pipeline_after_hog (list[dict]): A list of dict, where each element
            represents an operation defined in `mmselfsup.datasets.pipelines`.
        hog_para (dict): Config dict for HogGenerator.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self,
                 data_source,
                 pipeline_before_hog,
                 pipeline_after_hog,
                 hog_para,
                 prefetch=False):
        super().__init__(data_source, pipeline_before_hog, prefetch)
        self.gt_labels = self.data_source.get_gt_labels()
        pipeline_after_hog = [
            build_from_cfg(p, PIPELINES) for p in pipeline_after_hog
        ]
        self.pipeline_after_hog = Compose(pipeline_after_hog)
        self.hog_generator = HogGenerator(**hog_para)

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        img = self.data_source.get_img(idx)
        img = self.pipeline(img)
        hog = self.hog_generator(to_numpy(img))
        hog = torch.from_numpy(hog)
        img = self.pipeline_after_hog(img)

        return dict(img=img, label=label, idx=idx, hog=hog)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        eval_res = {}
        for name, val in results.items():
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.data_source.get_gt_labels())
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')

            num = val.size(0)
            _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                acc = correct_k * 100.0 / num
                eval_res[f'{name}_top{k}'] = acc
                if logger is not None and logger != 'silent':
                    print_log(f'{name}_top{k}: {acc:.03f}', logger=logger)
        return eval_res
