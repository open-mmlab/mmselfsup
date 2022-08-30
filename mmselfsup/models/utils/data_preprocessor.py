# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from mmengine.model import ImgDataPreprocessor

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
