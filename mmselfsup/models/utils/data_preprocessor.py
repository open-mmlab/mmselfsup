# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor

from mmselfsup.registry import MODELS


@MODELS.register_module()
class SelfSupDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for operations, like normalization and bgr to rgb.

    Compared with the :class:`mmengine.ImgDataPreprocessor`, this module treats
    each item in `inputs` of input data as a list, instead of torch.Tensor.
    """

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        assert isinstance(data,
                          dict), 'Please use default_collate in dataloader, \
            instead of pseudo_collate.'

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)
        # channel transform
        if self._channel_conversion:
            batch_inputs = [
                _input[:, [2, 1, 0], ...] for _input in batch_inputs
            ]

        # Convert to float after channel conversion to ensure
        # efficiency
        batch_inputs = [input_.float() for input_ in batch_inputs]

        # Normalization. Here is what is different from
        # :class:`mmengine.ImgDataPreprocessor`. Since there are multiple views
        # for an image for some  algorithms, e.g. SimCLR, each item in inputs
        # is a list, containing multi-views for an image.
        if self._enable_normalize:
            batch_inputs = [(_input - self.mean) / self.std
                            for _input in batch_inputs]

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class RelativeLocDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for Relative Location."""

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        batch_inputs, batch_data_samples = super().forward(data, training)
        # This part is unique to Relative Loc
        img1 = torch.stack(batch_inputs[1:], 1)  # Nx8xCxHxW
        img1 = img1.view(
            img1.size(0) * img1.size(1), img1.size(2), img1.size(3),
            img1.size(4))  # (8N)xCxHxW
        img2 = torch.unsqueeze(batch_inputs[0], 1).repeat(1, 8, 1, 1,
                                                          1)  # Nx8xCxHxW
        img2 = img2.view(
            img2.size(0) * img2.size(1), img2.size(2), img2.size(3),
            img2.size(4))  # (8N)xCxHxW
        batch_inputs = [img1, img2]

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class RotationPredDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for Relative Location."""

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        batch_inputs, batch_data_samples = super().forward(data, training)

        # This part is unique to Rotation Pred
        img = torch.stack(batch_inputs, 1)  # Nx4xCxHxW
        img = img.view(
            img.size(0) * img.size(1), img.size(2), img.size(3),
            img.size(4))  # (4N)xCxHxW
        batch_inputs = [img]

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class CAEDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for CAE.

    Compared with the :class:`mmselfsup.SelfSupDataPreprocessor`, this module
    will normalize the prediction image and target image with different
    normalization parameters.
    """

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)
        # channel transform
        if self._channel_conversion:
            batch_inputs = [
                _input[:, [2, 1, 0], ...] for _input in batch_inputs
            ]

        # Convert to float after channel conversion to ensure
        # efficiency
        batch_inputs = [input_.float() for input_ in batch_inputs]

        # Normalization. Here is what is different from
        # :class:`mmselfsup.SelfSupDataPreprocessor`. Normalize the target
        # image and prediction image with different normalization params
        if self._enable_normalize:
            batch_inputs = [(batch_inputs[0] - self.mean) / self.std,
                            batch_inputs[1] / 255. * 0.8 + 0.1]

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class TwoNormDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for CAE, BEiT v1/v2, etc.

    Compared with the :class:`mmselfsup.SelfSupDataPreprocessor`, this module
    will normalize the prediction image and target image with different
    normalization parameters.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``bgr_to_rgb=True`` it means the mean value of R,
            G, B channels. If the length of `mean` is 1, it means all
            channels have the same mean value, or the input is a gray image.
            If it is not specified, images will not be normalized. Defaults
            None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``bgr_to_rgb=True`` it means the standard
            deviation of R, G, B channels. If the length of `std` is 1,
            it means all channels have the same standard deviation, or the
            input is a gray image.  If it is not specified, images will
            not be normalized. Defaults None.
        second_mean (Sequence[float or int], optional): The description is
            like ``mean``, it can be customized for targe image. Defaults None.
        second_std (Sequence[float or int], optional): The description is
            like ``std``, it can be customized for targe image. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 second_mean: Sequence[Union[float, int]] = None,
                 second_std: Sequence[Union[float, int]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        assert (second_mean is not None) and (second_std is not None), (
            'mean and std should not be None while using '
            '`TwoNormDataPreprocessor`')
        assert len(second_mean) == 3 or len(second_mean) == 1, (
            '`mean` should have 1 or 3 values, to be compatible with '
            f'RGB or gray image, but got {len(second_mean)} values')
        assert len(second_std) == 3 or len(second_std) == 1, (
            '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
            f'or gray image, but got {len(std)} values')  # type: ignore

        self.register_buffer('second_mean',
                             torch.tensor(second_mean).view(-1, 1, 1), False)
        self.register_buffer('second_std',
                             torch.tensor(second_std).view(-1, 1, 1), False)

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
                model input.
        """
        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)
        # channel transform
        if self._channel_conversion:
            batch_inputs = [
                _input[:, [2, 1, 0], ...] for _input in batch_inputs
            ]

        # Convert to float after channel conversion to ensure
        # efficiency
        batch_inputs = [input_.float() for input_ in batch_inputs]

        # Normalization. Here is what is different from
        # :class:`mmselfsup.SelfSupDataPreprocessor`. Normalize the target
        # image and prediction image with different normalization params
        if self._enable_normalize:
            batch_inputs = [
                (batch_inputs[0] - self.mean) / self.std,
                (batch_inputs[1] - self.second_mean) / self.second_std
            ]

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class VideoDataPreprocessor(BaseDataPreprocessor):
    """Video pre-processor for operations, like normalization and bgr to rgb
    conversion .

    Compared with the :class:`mmaction.ActionDataPreprocessor`, this module
    treats each item in `inputs` of input data as a list, instead of
    torch.Tensor.

    Args:
        mean (Sequence[float or int, optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.bgr_to_rgb = bgr_to_rgb
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape == 'NCTHW':
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[List[torch.Tensor], Optional[list]]: Data in the same format
                as the model input.
        """

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)

        # ------ To RGB ------
        if self.bgr_to_rgb:
            if self.format_shape == 'NCHW':
                batch_inputs = [
                    batch_input[..., [2, 1, 0], :, :]
                    for batch_input in batch_inputs
                ]
            elif self.format_shape == 'NCTHW':
                batch_inputs = [
                    batch_input[..., [2, 1, 0], :, :, :]
                    for batch_input in batch_inputs
                ]
            else:
                raise ValueError(f'Invalid format shape: {self.format_shape}')

        # -- Normalization ---
        if self._enable_normalize:
            batch_inputs = [(batch_input - self.mean) / self.std
                            for batch_input in batch_inputs]
        else:
            batch_inputs = [
                batch_input.to(torch.float32) for batch_input in batch_inputs
            ]

        return batch_inputs, batch_data_samples
