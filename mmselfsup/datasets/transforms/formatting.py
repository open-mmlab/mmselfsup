# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, LabelData

from mmselfsup.registry import TRANSFORMS
from mmselfsup.structures import SelfSupDataSample


@TRANSFORMS.register_module()
class PackSelfSupInputs(BaseTransform):
    """Pack data into the format compatible with the inputs of algorithm.

    Required Keys:

    - img

    Added Keys:

    - data_samples
    - inputs

    Args:
        key (str): The key of image inputted into the model. Defaults to 'img'.
        algorithm_keys (List[str]): Keys of elements related
            to algorithms, e.g. mask. Defaults to [].
        pseudo_label_keys (List[str]): Keys set to be the attributes of
            pseudo_label. Defaults to [].
        meta_keys (List[str]): The keys of meta info of an image.
            Defaults to [].
    """

    def __init__(self,
                 key: str = 'img',
                 algorithm_keys: List[str] = [],
                 pseudo_label_keys: List[str] = [],
                 meta_keys: List[str] = []) -> None:
        assert isinstance(key, str), f'key should be the type of str, instead \
            of {type(key)}.'

        self.key = key
        self.algorithm_keys = algorithm_keys
        self.pseudo_label_keys = pseudo_label_keys
        self.meta_keys = meta_keys

    def transform(self,
                  results: Dict) -> Dict[torch.Tensor, SelfSupDataSample]:
        """Method to pack the data.

        Args:
            results (Dict): Result dict from the data pipeline.

        Returns:
            Dict:

            - 'inputs' (List[torch.Tensor]): The forward data of models.
            - 'data_samples' (SelfSupDataSample): The annotation info of the
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
        if len(self.pseudo_label_keys) > 0:
            pseudo_label = InstanceData()
            data_sample.pseudo_label = pseudo_label

        # gt_label, sample_idx, mask, pred_label will be set here
        for key in self.algorithm_keys:
            self.set_algorithm_keys(data_sample, key, results)

        # keys, except for gt_label, sample_idx, mask, pred_label, will be
        # set as the attributes of pseudo_label
        for key in self.pseudo_label_keys:
            # convert data to torch.Tensor
            value = to_tensor(results[key])
            setattr(data_sample.pseudo_label, key, value)

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    @classmethod
    def set_algorithm_keys(self, data_sample: SelfSupDataSample, key: str,
                           results: dict) -> None:
        """Set the algorithm keys of SelfSupDataSample.

        Args:
            data_sample (SelfSupDataSample): An instance of SelfSupDataSample.
            key (str): The key, which may be used by the algorithm, such as
                gt_label, sample_idx, mask, pred_label. For more keys, please
                refer to the attribute of SelfSupDataSample.
            results (dict): The results from the data pipeline.
        """
        value = to_tensor(results[key])
        if key == 'sample_idx':
            sample_idx = InstanceData(value=value)
            setattr(data_sample, 'sample_idx', sample_idx)
        elif key == 'mask':
            mask = InstanceData(value=value)
            setattr(data_sample, 'mask', mask)
        elif key == 'gt_label':
            gt_label = LabelData(value=value)
            setattr(data_sample, 'gt_label', gt_label)
        elif key == 'pred_label':
            pred_label = LabelData(value=value)
            setattr(data_sample, 'pred_label', pred_label)
        else:
            raise AttributeError(f'{key} is not a attribute of \
                SelfSupDataSample')

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f'(keys={self.key}, \
            algorithm_keys={self.algorithm_keys}, \
            pseudo_label_keys={self.pseudo_label_keys}, \
            meta_keys={self.meta_keys})')
