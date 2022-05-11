# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform

from mmselfsup.core.data_structures import SelfSupDataSample
from mmselfsup.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackSelfSupInputs(BaseTransform):
    """Pack data into the format compatible with the inputs of algorithm.

    Required Keys:

    - img

    Added Keys:

    - data_sample
    - inputs

    Args:
        key (str): The key of image inputted into the model. Defaults to 'img'.
        training_related_keys (List[str]): Keys of elements related
            to the training, e.g. mask. Defaults to [].
        meta_keys (List[str]): The keys of meta info of an image.
            Defaults to [].
    """

    def __init__(self,
                 key: Optional[str] = 'img',
                 training_related_keys: Optional[List[str]] = [],
                 meta_keys: Optional[List[str]] = []) -> None:
        self.key = key
        self.training_related_keys = training_related_keys
        self.meta_keys = meta_keys

    def transform(self,
                  results: Dict) -> Dict[torch.Tensor, SelfSupDataSample]:
        """Method to pack the data.

        Args:
            results (Dict): Result dict from the data pipeline.

        Returns:
            Dict:

            - 'inputs' (List[torch.Tensor]): The forward data of models.
            - 'data_sample' (SelfSupDataSample): The annotation info of the
                the forward data.
        """
        packed_results = dict()
        if self.key in results:
            img = results[self.key]
            # if img is not a list, convert it to a list
            if not isinstance(img, List):
                img = [img]
            for i, img_ in enumerate(img):
                if len(img_.shape) < 3:
                    img_ = np.expand_dims(img_, -1)
                img_ = np.ascontiguousarray(img_.transpose(2, 0, 1))
                img[i] = to_tensor(img_)
            packed_results['inputs'] = img

        data_sample = SelfSupDataSample()

        for key in self.training_related_keys:
            setattr(data_sample, key, results[key])

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f'(keys={self.key}, \
            training_related_keys={self.training_related_keys}, \
            meta_keys={self.meta_keys})')
