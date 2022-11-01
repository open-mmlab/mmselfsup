# Copyright (c) OpenMMLab. All rights reserved.
import math
import numbers
import random
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image import (adjust_brightness, adjust_color, adjust_contrast,
                        adjust_hue)
from mmcv.transforms import BaseTransform
from PIL import Image, ImageFilter

from mmselfsup.registry import TRANSFORMS


def check_sequence_input(x: Sequence, name: str, req_sizes: tuple) -> None:
    """Check if the input is a sequence with the required sizes.

    Args:
        x (Sequence): The input sequence.
        name (str): The name of the input.
        req_sizes (tuple): The required sizes of the input.

    Returns:
        None
    """
    msg = req_sizes[0] if len(req_sizes) < 2 else ' or '.join(
        [str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError('{} should be a sequence of length {}.'.format(
            name, msg))
    if len(x) not in req_sizes:
        raise ValueError('{} should be sequence of length {}.'.format(
            name, msg))


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

    def transform(self, results: dict) -> dict:
        """Method to generate random block mask for each Image in SimMIM.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with added key ``mask``.
        """
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

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
        """Get the shape of mask.

        Returns:
            Tuple[int, int]: The shape of mask.
        """
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        """Generate mask recursively.

        Args:
            mask (np.ndarray): The mask to be generated.
            max_mask_patches (int): The maximum number of patches to be masked.

        Returns:
            int: The number of patches masked.
        """
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

    def transform(self, results: dict) -> dict:
        """Method to generate random block mask for each Image in BEiT.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with added key ``mask``.
        """
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
        img_h, img_w = img.shape[:2]
        area = img_h * img_w

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img_w and h <= img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img_w / img_h
        if in_ratio < min(ratio):
            w = img_w
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img_h
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img_w
            h = img_h
        i = (img_h - h) // 2
        j = (img_w - w) // 2
        return i, j, h, w

    def transform(self, results: dict) -> dict:
        """Crop the given image and resize it to two different sizes.

        This module crops the given image randomly and resize the crop to two
        different sizes. This is popularly used in BEiT-style masked image
        modeling, where an off-the-shelf model is used to provide the target.

        Args:
            results (dict): Results from previous pipeline.

        Returns:
            dict: Results after applying this transformation.
        """
        img = results['img']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            img = img[i:i + h, j:j + w]
            img = cv2.resize(img, self.size, interpolation=interpolation)
            results.update({'img': img})

        else:
            img = img[i:i + h, j:j + w]
            img_sample = cv2.resize(
                img, self.size, interpolation=interpolation)
            img_target = cv2.resize(
                img, self.second_size, interpolation=self.second_interpolation)
            results.update({'img': [img_sample, img_target]})
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

    def transform(self, results: dict) -> dict:
        """Apply GaussianBlur augmentation to the given image.

        Args:
            results (dict): Results from previous pipeline.

        Returns:
            dict: Results after applying this transformation.
        """
        if np.random.rand() > self.prob:
            return results
        img_pil = Image.fromarray(results['img'].astype('uint8'))
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

    def __init__(self, threshold: int = 128, prob: float = 0.5) -> None:
        super().__init__()
        assert 0 <= prob <= 1.0, \
            f'The prob should be in range [0, 1], got {prob} instead.'

        self.threshold = threshold
        self.prob = prob

    def transform(self, results: dict) -> dict:
        """Apply Solarize augmentation to the given image.

        Args:
            results (dict): Results from previous pipeline.

        Returns:
            dict: Results after applying this transformation.
        """
        if np.random.rand() > self.prob:
            return results
        img = results['img']
        results['img'] = mmcv.solarize(img, thr=self.threshold)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RotationWithLabels(BaseTransform):
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

    def _rotate(self, img: np.ndarray) -> List[np.ndarray]:
        """Rotate input image with 0, 90, 180, and 270 degrees.

        Args:
            img (np.ndarray: input image of shape (H, W, C).

        Returns:
            List[np.ndarray]: A list of four rotated images.
        """
        return [
            img,
            mmcv.imrotate(img, 90),
            mmcv.imrotate(img, 180),
            mmcv.imrotate(img, 270),
        ]

    def transform(self, results: dict) -> dict:
        """Apply rotation augmentation to the given image.

        Args:
            results (dict): Results from previous pipeline.

        Returns:
            dict: Results after applying this transformation.
        """
        img = self._rotate(results['img'])
        rotation_labels = np.array([0, 1, 2, 3])
        results['img'] = img
        results['rot_label'] = rotation_labels
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
    - patch_box
    - unpatched_img

    Crops image into several patches and concatenates every surrounding patch
    with center one. Finally gives labels `0, 1, 2, 3, 4, 5, 6, 7` and patch
    positions.
    """

    def __init__(self) -> None:
        pass

    def _image_to_patches(self, img: np.ndarray) -> List[np.ndarray]:
        """Crop split_per_side x split_per_side patches from input image.

        Args:
            img (np.ndarray): input image.

        Returns:
            patches (List[np.ndarray]): A list of cropped patches.
        """
        split_per_side = 3  # split of patches per image side
        patch_jitter = 21  # jitter of each patch from each grid
        h, w, _ = img.shape
        h_grid = h // split_per_side
        w_grid = w // split_per_side
        h_patch = h_grid - patch_jitter
        w_patch = w_grid - patch_jitter
        assert h_patch > 0 and w_patch > 0
        patches = []
        patches_pos = []
        for i in range(split_per_side):
            for j in range(split_per_side):
                # get a patch of the image
                patch_box = np.array([
                    i * h_grid, j * w_grid, (i + 1) * h_grid, (j + 1) * w_grid
                ])
                patch = mmcv.imcrop(img, bboxes=patch_box)

                # random crop sub-patch in the patch
                ymin, xmin, height, width = RandomCrop.get_params(
                    patch, (h_patch, w_patch))
                patch = mmcv.imcrop(
                    patch,
                    np.array([
                        xmin,
                        ymin,
                        xmin + width - 1,
                        ymin + height - 1,
                    ]))
                patches.append(patch)
                patches_pos.append(
                    np.array([
                        i * h_grid + xmin,
                        j * w_grid + ymin,
                        i * h_grid + xmin + width - 1,
                        j * w_grid + ymin + height - 1,
                    ]))
        return patches, patches_pos

    def transform(self, results: dict) -> dict:
        """Apply random patch augmentation to the given image.

        Args:
            results (dict): Results from previous pipeline.

        Returns:
            dict: Results after applying this transformation.
        """
        img = results['img']
        patches, patches_pos = self._image_to_patches(img)
        patches_pos = np.stack(patches_pos, axis=0)

        multi_views = []
        multi_views.append(patches[4])
        for i in range(9):
            if i != 4:
                multi_views.append(patches[i])

        # create corresponding labels for patch pairs
        patch_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        results['img'] = multi_views  # 8HWC
        results['patch_label'] = np.expand_dims(patch_labels, axis=0)
        results['patch_box'] = np.expand_dims(patches_pos, axis=0)
        results['unpatched_img'] = np.expand_dims(img, axis=0)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast, saturation and hue of an
    image.

    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter
            brightness. brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter
            contrast. contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
            Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter
            saturation. saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
            Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given
            [min, max]. Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be
            non-negative for conversion to HSV space; thus it does not work
            if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using
            this function.
        backend (str): The type of image processing backend. Options are
            `cv2`, `pillow`. Defaults to `pillow`.
    """  # noqa: E501

    def __init__(self,
                 brightness: Union[float, List[float]] = 0,
                 contrast: Union[float, List[float]] = 0,
                 saturation: Union[float, List[float]] = 0,
                 hue: Union[float, List[float]] = 0,
                 backend: str = 'pillow') -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(
            hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.backend = backend

    def _check_input(
            self,
            value: float,
            name: str,
            center: float = 1.,
            bound: Tuple = (0, float('inf')),
            clip_first_on_zero: bool = True) -> Union[List[float], None]:
        """Check the input and convert it to the tuple format.

        Args:
            value (float or list of float): The input value.
            name (str): The name of the input.
            center (float): The center value of the input.
            bound (tuple of float): The bound of the input.
            clip_first_on_zero (bool): Whether to clip the first value on zero.

        Returns:
            Union[List[float], None]: The converted value or None.
        """

        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')
        else:
            raise TypeError(
                f'{name} should be a single number or a tuple with length 2.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[float],
               Optional[float]]:
        """Get the parameters for the randomized transform to be applied on
        image.

        Args:
            brightness (tuple of float (min, max), optional): The range from
                which the brightness_factor is chosen uniformly. Pass None to
                turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from
                which the contrast_factor is chosen uniformly. Pass None to
                turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from
                which the saturation_factor is chosen uniformly. Pass None to
                turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the
                hue_factor is chosen uniformly. Pass None to turn off the
                transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
                along with their random order.
        """
        order = np.random.permutation(4)

        b = None if brightness is None else float(
            np.random.uniform(brightness[0], brightness[1]))
        c = None if contrast is None else float(
            np.random.uniform(contrast[0], contrast[1]))
        s = None if saturation is None else float(
            np.random.uniform(saturation[0], saturation[1]))
        h = None if hue is None else float(np.random.uniform(hue[0], hue[1]))

        return b, c, s, h, order

    def transform(self, results: dict) -> dict:
        """Randomly change the brightness, contrast, saturation and hue of an
        image. # noqa: E501.

        Args:
            results (dict): The results dict from previous pipeline.

        Returns:
            dict: Results after applying this transformation.
        """
        brightness_factor, contrast_factor, saturation_factor, hue_factor, \
            order = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

        img = results['img'].astype('uint8')
        for fn_id in order:
            if fn_id == 0 and brightness_factor is not None:
                img = adjust_brightness(
                    img, brightness_factor, backend=self.backend)
            elif fn_id == 1 and contrast_factor is not None:
                img = adjust_contrast(
                    img, contrast_factor, backend=self.backend)
            elif fn_id == 2 and saturation_factor is not None:
                img = adjust_color(
                    img, saturation_factor, backend=self.backend)
            elif fn_id == 3 and hue_factor is not None:
                img = adjust_hue(img, hue_factor, backend=self.backend)
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation},'
        repr_str += f'saturation={self.hue})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(BaseTransform):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        size (Sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (Tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (Tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size: Union[int, Sequence[int]],
                 scale: Tuple = (0.08, 1.0),
                 ratio: Tuple = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        super().__init__()
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img: np.ndarray,
                   scale: Tuple,
                   ratio: Tuple,
                   max_attempts: int = 10) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarray): Image to be cropped.
            scale (Tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (Tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    def transform(self, results: dict) -> dict:
        """Randomly crop the image and resize the image to the target size.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        """
        img = results['img']
        get_params_args = dict(
            img=img,
            scale=self.scale,
            ratio=self.ratio,
            max_attempts=self.max_attempts)
        ymin, xmin, ymax, xmax = self.get_params(**get_params_args)
        img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
        results['img'] = mmcv.imresize(
            img,
            tuple(self.size[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Crop the given Image at a random location.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        size (int or Sequence): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or Sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:

            - constant: Pads with a constant value, this value is specified \
                with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: Pads with reflection of image without repeating the \
                last value on the edge. For example, padding [1, 2, 3, 4] \
                with 2 elements on both sides in reflect mode will result \
                in [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: Pads with reflection of image repeating the last \
                value on the edge. For example, padding [1, 2, 3, 4] with \
                2 elements on both sides in symmetric mode will result in \
                [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 size: Union[int, Sequence[int]],
                 padding: Optional[Union[int, Sequence[int]]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[numbers.Number, Sequence[numbers.Number]] = 0,
                 padding_mode: str = 'constant') -> None:
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img: np.ndarray, output_size: Tuple) -> Tuple:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (np.ndarray): Image to be cropped.
            output_size (Tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        ymin = random.randint(0, height - target_height)
        xmin = random.randint(0, width - target_width)
        return ymin, xmin, target_height, target_width

    def transform(self, results: dict) -> dict:
        """Randomly crop the image.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        """
        img = results['img']
        if self.padding is not None:
            img = mmcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = mmcv.impad(
                img,
                padding=(0, self.size[0] - img.shape[0], 0,
                         self.size[0] - img.shape[0]),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = mmcv.impad(
                img,
                padding=(self.size[1] - img.shape[1], 0,
                         self.size[1] - img.shape[1], 0),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        ymin, xmin, height, width = self.get_params(img, self.size)
        results['img'] = mmcv.imcrop(
            img, np.array([
                xmin,
                ymin,
                xmin + width - 1,
                ymin + height - 1,
            ]))
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(size={self.size}, padding={self.padding})')


@TRANSFORMS.register_module()
class RandomRotation(BaseTransform):
    """Rotate the image by angle.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        degrees (sequence | int): Range of degrees to select from. If degrees
            is an int instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (str, optional): Interpolation method, accepted values
            are 'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults
            to 'nearest'.
        expand (bool, optional): Optional expansion flag. If true, expands the
            output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the
            input image. Note that the expand flag assumes rotation around the
            center and no translation. Defaults to False.
        center (Tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Defaults to None.
        fill (int, optional): Pixel fill value for the area outside the rotated
            image. Default to 0.
    """

    def __init__(self,
                 degrees: Union[int, Sequence[int]],
                 interpolation: str = 'nearest',
                 expand: bool = False,
                 center: Optional[Tuple[float]] = None,
                 fill: int = 0) -> None:
        super().__init__()

        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')

        if not isinstance(expand, bool):
            raise TypeError('expand should be bool.')

        if center is not None:
            check_sequence_input(center, 'center', req_sizes=(2, ))
            if expand:
                raise ValueError('`expand` conflicts with `center`')

        if not isinstance(fill, int):
            raise TypeError('fill should be int.')

        degrees = self._setup_angle(degrees, name='degrees', req_sizes=(2, ))

        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def _setup_angle(self,
                     x: Union[int, Sequence[int]],
                     name: str,
                     req_sizes: Tuple = (2, )) -> List[float]:
        """Setup the angle.

        Args:
            x (Union[int, Sequence[int]]): Range of degrees to select from. If
                degrees is an int instead of sequence like (min, max), the
                range of degrees will be (-degrees, +degrees).
            name (str): Name of the angle.
            req_sizes (Tuple): Required sizes of the angle.

        Returns:
            List[float]: The range of valid angles.
        """
        if isinstance(x, int):
            if x < 0:
                raise ValueError(
                    'If {} is a single number, it must be positive.'.format(
                        name))
            x = [-x, x]
        else:
            check_sequence_input(x, name, req_sizes)

        return [float(d) for d in x]

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Args:
            degrees (List[float]): Range of degrees to select from.

        Returns:
            float: angle parameter to be passed to ``rotate`` for
                random rotation.
        """
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]),
                                    float(degrees[1])).item())
        return angle

    def transform(self, results: dict) -> dict:
        """Randomly rotate the image.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        """
        img = results['img']
        angle = self.get_params(self.degrees)
        results['img'] = mmcv.imrotate(
            img,
            angle,
            center=self.center,
            border_value=self.fill,
            interpolation=self.interpolation,
            auto_bound=self.expand)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(degrees={self.degrees}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', expand={self.expand}'
        if self.center is not None:
            repr_str += f', center={self.center}'
        if self.fill is not None:
            repr_str += f', fill={self.fill})'
        return repr_str
