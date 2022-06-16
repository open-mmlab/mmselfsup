# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
from mmengine.data import BaseDataElement
from mmengine.model import ImgDataPreprocessor

from mmselfsup.registry import MODELS


@MODELS.register_module()
class SelfSupDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for operations, like normalization and bgr to rgb.

    Compared with the :class:`mmengine.ImgDataPreprocessor`, this module treats
    each item in `inputs` of input data as a list, instead of torch.Tensor.
    """

    def collate_data(
            self,
            data: Sequence[dict]) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Collating and copying data to the target device.

        This module overwrite the default method by treating each item in
        ``input`` of the input data as a list.

        Collates the data sampled from dataloader into a list of tensor and
        list of labels, and then copies tensor to the target device.

        Subclasses could override it to be compatible with the custom format
        data sampled from custom dataloader.

        Args:
            data (Sequence[dict]): Data sampled from dataloader.

        Returns:
            Tuple[List[torch.Tensor], Optional[list]]: Unstacked list of input
            tensor and list of labels at target device.
        """
        inputs = [[img.to(self.device) for img in _data['inputs']]
                  for _data in data]
        batch_data_samples: List[BaseDataElement] = []
        # Model can get predictions without any data samples.
        for _data in data:
            if 'data_sample' in _data:
                batch_data_samples.append(_data['data_sample'])
        # Move data from CPU to corresponding device.
        batch_data_samples = [
            data_sample.to(self.device) for data_sample in batch_data_samples
        ]

        if not batch_data_samples:
            batch_data_samples = None  # type: ignore

        return inputs, batch_data_samples

    def forward(
            self,
            data: Sequence[dict],
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        inputs, batch_data_samples = self.collate_data(data)
        # channel transform
        if self.channel_conversion:
            inputs = [[img_[[2, 1, 0], ...] for img_ in _input]
                      for _input in inputs]

        # Normalization. Here is what is different from
        # :class:`mmengine.ImgDataPreprocessor`. Since there are multiple views
        # for an image for some  algorithms, e.g. SimCLR, each item in inputs
        # is a list, containing multi-views for an image.
        inputs = [[(img_ - self.mean) / self.std for img_ in _input]
                  for _input in inputs]

        batch_inputs = []
        for i in range(len(inputs[0])):
            cur_batch = [img[i] for img in inputs]
            batch_inputs.append(torch.stack(cur_batch))

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class RelativeLocDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for Relative Location."""

    def forward(
            self,
            data: Sequence[dict],
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        inputs, batch_data_samples = self.collate_data(data)
        # channel transform
        if self.channel_conversion:
            inputs = [[img_[[2, 1, 0], ...] for img_ in _input]
                      for _input in inputs]

        # Normalization. Here is what is different from
        # :class:`mmengine.ImgDataPreprocessor`. Since there are multiple views
        # for an image for some  algorithms, e.g. SimCLR, each item in inputs
        # is a list, containing multi-views for an image.
        inputs = [[(img_ - self.mean) / self.std for img_ in _input]
                  for _input in inputs]

        batch_inputs = []
        for i in range(len(inputs[0])):
            cur_batch = [img[i] for img in inputs]
            batch_inputs.append(torch.stack(cur_batch))

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
