# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch

from mmselfsup.registry import MODELS
from .data_preprocessor import SelfSupDataPreprocessor


@MODELS.register_module()
class CAEDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for CAE.

    Compared with the :class:`mmselfsup.SelfSupDataPreprocessor`, this module
    will normalize the prediction image and target image with different
    normalization parameters.
    """

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
        # :class:`mmselfsup.SelfSupDataPreprocessor`. Normalize the target
        # image and prediction image with different normalization params
        inputs = [[(_input[0] - self.mean) / self.std,
                   _input[1] / 255. * 0.8 + 0.1] for _input in inputs]

        batch_inputs = []
        for i in range(len(inputs[0])):
            cur_batch = [img[i] for img in inputs]
            batch_inputs.append(torch.stack(cur_batch))

        return batch_inputs, batch_data_samples
