# Copyright (c) OpenMMLab. All rights reserved.
# TODO: will use real PixelData once it is added in mmengine
from mmengine.data import BaseDataElement
from mmengine.data import BaseDataElement as PixelData
from mmengine.data import InstanceData


class SelfSupDataSample(BaseDataElement):
    """A data structure interface of MMSelfSup. They are used as interfaces
    between different components.

    The attributes in ``SelfSupDataSample`` are divided into several parts:

        - ``gt_label``(InstanceData): The ground truth label of an image.
        - ``idx``(InstanceData): The idx of an image in the dataset.
        - ``mask``(PixelData): Mask used in masks image modeling.
        - ``pred_label``(InstanceData): Label used in pretext task,
        e.g. Relative Location.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.data import InstanceData
         >>> from mmselfsup.core import SelfSupDataSample

         >>> data_sample = SelfSupDataSample()
         >>> gt_label = InstanceData()
         >>> gt_label.value = [1]
         >>> data_sample.gt_label = gt_label
         >>> len(data_sample.gt_label)
         1
         >>> print(data_sample)
        <SelfSupDataSample(

            META INFORMATION

            DATA FIELDS
            gt_label: <InstanceData(

                    META INFORMATION

                    DATA FIELDS
                    value: [1]
                ) at 0x7f15c08f9d10>
            _gt_label: <InstanceData(

                    META INFORMATION

                    DATA FIELDS
                    value: [1]
                ) at 0x7f15c08f9d10>
         ) at 0x7f15c077ef10>
         >>> idx = InstanceData()
         >>> idx.value = [0]
         >>> data_sample = SelfSupDataSample(idx=idx)
         >>> assert 'idx' in data_sample

         >>> data_sample = SelfSupDataSample()
         >>> mask = dict(value=np.random.rand(48, 48))
         >>> mask = PixelData(**mask)
         >>> data_sample.mask = mask
         >>> assert 'mask' in data_sample
         >>> assert 'value' in data_sample.mask

         >>> data_sample = SelfSupDataSample()
         >>> pred_label = dict(pred_label=[3])
         >>> pred_label = InstanceData(**pred_label)
         >>> data_sample.pred_label = pred_label
         >>> print(data_sample)
        <SelfSupDataSample(

            META INFORMATION

            DATA FIELDS
            _pred_label: <InstanceData(

                    META INFORMATION

                    DATA FIELDS
                    pred_label: [3]
                ) at 0x7f15c06a3990>
            pred_label: <InstanceData(

                    META INFORMATION

                    DATA FIELDS
                    pred_label: [3]
                ) at 0x7f15c06a3990>
        ) at 0x7f15c07b8bd0>
    """

    @property
    def gt_label(self) -> InstanceData:
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: InstanceData):
        self.set_field(value, '_gt_label', dtype=InstanceData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

    @property
    def idx(self) -> InstanceData:
        return self._idx

    @idx.setter
    def idx(self, value: InstanceData):
        self.set_field(value, '_idx', dtype=InstanceData)

    @idx.deleter
    def idx(self):
        del self._idx

    @property
    def mask(self) -> PixelData:
        return self._mask

    @mask.setter
    def mask(self, value: PixelData):
        self.set_field(value, '_mask', dtype=PixelData)

    @mask.deleter
    def mask(self):
        del self._mask

    @property
    def pred_label(self) -> InstanceData:
        return self._pred_label

    @pred_label.setter
    def pred_label(self, value: InstanceData):
        self.set_field(value, '_pred_label', dtype=InstanceData)

    @pred_label.deleter
    def pred_label(self):
        del self._pred_label
