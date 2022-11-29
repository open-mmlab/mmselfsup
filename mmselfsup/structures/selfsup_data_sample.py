# Copyright (c) OpenMMLab. All rights reserved.
# TODO: will use real PixelData once it is added in mmengine
from mmengine.structures import BaseDataElement, InstanceData, LabelData


class SelfSupDataSample(BaseDataElement):
    """A data structure interface of MMSelfSup. They are used as interfaces
    between different components.

    Meta field:

      - ``img_shape`` (Tuple): The shape of the corresponding input image.
        Used for visualization.

      - ``ori_shape`` (Tuple): The original shape of the corresponding image.
        Used for visualization.

      - ``img_path`` (str): The path of original image.

    Data field:

      - ``gt_label`` (LabelData): The ground truth label of an image.

      - ``sample_idx`` (InstanceData): The idx of an image in the dataset.

      - ``mask`` (BaseDataElement): Mask used in masks image modeling.

      - ``pred_label`` (LabelData): The predicted label.

      - ``pseudo_label`` (InstanceData): Label used in pretext task,
        e.g. Relative Location.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from mmengine.structure import InstanceData
        >>> from mmselfsup.structures import SelfSupDataSample

        >>> data_sample = SelfSupDataSample()
        >>> gt_label = LabelData()
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
        >>> pred_label = LabelData(**pred_label)
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
    def gt_label(self) -> LabelData:
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label

    @property
    def sample_idx(self) -> InstanceData:
        return self._sample_idx

    @sample_idx.setter
    def sample_idx(self, value: InstanceData):
        self.set_field(value, '_sample_idx', dtype=InstanceData)

    @sample_idx.deleter
    def sample_idx(self):
        del self._sample_idx

    @property
    def mask(self) -> BaseDataElement:
        return self._mask

    @mask.setter
    def mask(self, value: BaseDataElement):
        self.set_field(value, '_mask', dtype=BaseDataElement)

    @mask.deleter
    def mask(self):
        del self._mask

    @property
    def pred_label(self) -> LabelData:
        return self._pred_label

    @pred_label.setter
    def pred_label(self, value: LabelData):
        self.set_field(value, '_pred_label', dtype=LabelData)

    @pred_label.deleter
    def pred_label(self):
        del self._pred_label

    @property
    def pseudo_label(self) -> BaseDataElement:
        return self._pseudo_label

    @pseudo_label.setter
    def pseudo_label(self, value: BaseDataElement):
        self.set_field(value, '_pseudo_label', dtype=BaseDataElement)

    @pseudo_label.deleter
    def pseudo_label(self):
        del self._pseudo_label
