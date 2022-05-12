# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from mmcv.image import adjust_lighting, solarize
from mmcv.transforms import BaseTransform
from PIL import Image, ImageFilter
from timm.data import create_transform
from torchvision.transforms import RandomCrop

from mmselfsup.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SimMIMMaskGenerator(BaseTransform):
    """Generate random block mask for each Image.

    Added Keys:

    - mask

    This module is used in SimMIM to generate masks.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
    """

    def __init__(self,
                 input_size: int = 192,
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.6) -> None:
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def transform(self, results: Dict) -> Dict:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        mask = torch.from_numpy(mask)  # H X W

        results.update({'mask': mask})

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'mask_patch_size={self.mask_patch_size}, '
        repr_str += f'model_patch_size={self.model_patch_size}, '
        repr_str += f'mask_ratio={self.mask_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class BEiTMaskGenerator(BaseTransform):
    """Generate mask for image.

    Added Keys:

    - mask

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit

    Args:
        input_size (int): The size of input image.
        num_masking_patches (int): The number of patches to be masked.
        min_num_patches (int): The minimum number of patches to be masked
            in the process of generating mask. Defaults to 4.
        max_num_patches (int, optional): The maximum number of patches to be
            masked in the process of generating mask. Defaults to None.
        min_aspect (float): The minimum aspect ratio of mask blocks. Defaults
            to 0.3.
        min_aspect (float, optional): The minimum aspect ratio of mask blocks.
            Defaults to None.
    """

    def __init__(self,
                 input_size: int,
                 num_masking_patches: int,
                 min_num_patches: int = 4,
                 max_num_patches: Optional[int] = None,
                 min_aspect: float = 0.3,
                 max_aspect: Optional[float] = None) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None \
            else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches,
                                         max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top:top + h, left:left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def transform(self, results: Dict) -> Dict:
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count != self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            mask_count += delta
        results.update({'mask': mask})

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(height={self.height}, '
        repr_str += f'width={self.width}, '
        repr_str += f'num_patches={self.num_patches}, '
        repr_str += f'num_masking_patches={self.num_masking_patches}, '
        repr_str += f'min_num_patches={self.min_num_patches}, '
        repr_str += f'max_num_patches={self.max_num_patches}, '
        repr_str += f'log_aspect_ratio={self.log_aspect_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCropAndInterpolationWithTwoPic(BaseTransform):
    """Crop the given PIL Image to random size and aspect ratio with random
    interpolation.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - target_img

    This module is borrowed from
    https://github.com/microsoft/unilm/tree/master/beit.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size. This is popularly used
    to train the Inception networks. This module first crops the image and
    resizes the crop to two different sizes.

    Args:
        size (Union[tuple, int]): Expected output size of each edge of the
            first image.
        second_size (Union[tuple, int], optional): Expected output size of each
            edge of the second image.
        scale (tuple[float, float]): Range of size of the origin size cropped.
            Defaults to (0.08, 1.0).
        ratio (tuple[float, float]): Range of aspect ratio of the origin aspect
            ratio cropped. Defaults to (3./4., 4./3.).
        interpolation (str): The interpolation for the first image. Defaults
            to ``bilinear``.
        second_interpolation (str): The interpolation for the second image.
            Defaults to ``lanczos``.
    """

    interpolation_dict = {
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

    def __init__(self,
                 size: Union[tuple, int],
                 second_size=None,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 second_interpolation='lanczos') -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn('range should be of kind (min, max)')

        if interpolation == 'random':
            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self.interpolation_dict.get(
                interpolation, Image.BILINEAR)
        self.second_interpolation = self.interpolation_dict.get(
            second_interpolation, Image.BILINEAR)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: np.ndarray, scale: tuple,
                   ratio: tuple) -> Sequence[int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect
                ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def transform(self, results: Dict) -> Dict:
        img = results['img']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
            results.update({'img': img})

        else:
            img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
            img_target = F.resized_crop(img, i, j, h, w, self.second_size,
                                        self.second_interpolation)
            results.update({'img': img, 'target_img': img_target})
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'second_size={self.second_size}, '
        repr_str += f'interpolation={self.interpolation}, '
        repr_str += f'second_interpolation={self.second_interpolation}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'ratio={self.ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomAug(object):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation
    with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    This code is borrowed from <https://github.com/pengzhiliang/MAE-pytorch>
    """

    def __init__(self,
                 input_size=None,
                 color_jitter=None,
                 auto_augment=None,
                 interpolation=None,
                 re_prob=None,
                 re_mode=None,
                 re_count=None,
                 mean=None,
                 std=None):

        self.trans = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            mean=mean,
            std=std,
        )

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
@TRANSFORMS.register_module()
class Lighting(BaseTransform):
    """Adjust images lighting using AlexNet-style PCA jitter.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        eigval (list): the eigenvalue of the convariance matrix of pixel
            values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of pixel
            values, respectively.
        alphastd (float): The standard deviation for distribution of alpha.
            Defaults to 0.1
        to_rgb (bool): Whether to convert img to rgb.
    """

    def __init__(self,
                 eigval: Optional[List] = [0.2175, 0.0188, 0.0045],
                 eigvec: Optional[List[List]] = [
                     [-0.5675, 0.7192, 0.4009],
                     [-0.5808, -0.0045, -0.8140],
                     [-0.5836, -0.6948, 0.4203],
                 ],
                 alphastd: Optional[float] = 0.1,
                 to_rgb: Optional[bool] = True) -> None:
        assert isinstance(eigval, list), \
            f'eigval must be of type list, got {type(eigval)} instead.'
        assert isinstance(eigvec, list), \
            f'eigvec must be of type list, got {type(eigvec)} instead.'
        for vec in eigvec:
            assert isinstance(vec, list) and len(vec) == len(eigvec[0]), \
                'eigvec must contains lists with equal length.'
        self.eigval = np.array(eigval)
        self.eigvec = np.array(eigvec)
        self.alphastd = alphastd
        self.to_rgb = to_rgb

    def transform(self, results: Dict) -> Dict:
        img = results['img']
        results['img'] = adjust_lighting(
            img,
            self.eigval,
            self.eigvec,
            alphastd=self.alphastd,
            to_rgb=self.to_rgb)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(eigval={self.eigval.tolist()}, '
        repr_str += f'eigvec={self.eigvec.tolist()}, '
        repr_str += f'alphastd={self.alphastd}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class RandomGaussianBlur(BaseTransform):
    """GaussianBlur augmentation refers to `SimCLR.

    <https://arxiv.org/abs/2002.05709>`_.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
        prob (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self,
                 sigma_min: float,
                 sigma_max: float,
                 prob: Optional[float] = 0.5) -> None:
        super().__init__()
        assert 0 <= prob <= 1.0, \
            f'The prob should be in range [0,1], got {prob} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = prob

    def transform(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results
        img_pil = Image.fromarray(results['img'])
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img_pil_res = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        results['img'] = np.array(img_pil_res)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomSolarize(BaseTransform):
    """Solarization augmentation refers to `BYOL.

    <https://arxiv.org/abs/2006.07733>`_.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        threshold (float, optional): The solarization threshold.
            Defaults to 128.
        prob (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self,
                 threshold: Optional[int] = 128,
                 prob: Optional[float] = 0.5) -> None:
        super().__init__()
        assert 0 <= prob <= 1.0, \
            f'The prob should be in range [0, 1], got {prob} instead.'

        self.threshold = threshold
        self.prob = prob

    def transform(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results
        img = results['img']
        results['img'] = solarize(img, thr=self.threshold)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotationWithLabels(BaseTransform):
    """Rotation prediction.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - rot_label

    Rotate each image with 0, 90, 180, and 270 degrees and give labels `0, 1,
    2, 3` correspodingly.
    """

    def __init__(self) -> None:
        pass

    def _rotate(self, img: torch.Tensor):
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

    def transform(self, results: Dict) -> Dict:
        img = np.transpose(results['img'], (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.stack(self._rotate(img), dim=0)
        rotation_labels = np.array([0, 1, 2, 3])
        results = dict(img=img.numpy(), rot_label=rotation_labels)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class RandomPatchWithLabels(BaseTransform):
    """Relative patch location.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - patch_label

    Crops image into several patches and concatenates every surrounding patch
    with center one. Finally gives labels `0, 1, 2, 3, 4, 5, 6, 7`.
    """

    def __init__(self) -> None:
        pass

    def _image_to_patches(self, img: Image):
        """Crop split_per_side x split_per_side patches from input image.

        Args:
            img (PIL Image): input image.

        Returns:
            list[PIL Image]: A list of cropped patches.
        """
        split_per_side = 3  # split of patches per image side
        patch_jitter = 21  # jitter of each patch from each grid
        h, w = img.size
        h_grid = h // split_per_side
        w_grid = w // split_per_side
        h_patch = h_grid - patch_jitter
        w_patch = w_grid - patch_jitter
        assert h_patch > 0 and w_patch > 0
        patches = []
        for i in range(split_per_side):
            for j in range(split_per_side):
                p = F.crop(img, i * h_grid, j * w_grid, h_grid, w_grid)
                p = RandomCrop((h_patch, w_patch))(p)
                patches.append(np.transpose(np.asarray(p), (2, 0, 1)))
        return patches

    def transform(self, results: Dict) -> Dict:
        img = Image.fromarray(results['img'])
        patches = self._image_to_patches(img)
        perms = []
        # create a list of patch pairs
        [
            perms.append(np.concatenate((patches[i], patches[4]), axis=0))
            for i in range(9) if i != 4
        ]
        # create corresponding labels for patch pairs
        patch_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        results = dict(
            img=np.stack(perms, axis=0),
            patch_label=patch_labels)  # 8(2C)HW, 8
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
