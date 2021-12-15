# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module
class ImageList(BaseDataSource):
    """The implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.
    """

    def load_annotations(self):
        assert self.ann_file is not None
        if not isinstance(self.ann_file, list):
            self.ann_file = [self.ann_file]

        data_infos = []
        for ann_file in self.ann_file:
            with open(ann_file, 'r') as f:
                self.samples = f.readlines()
            self.has_labels = len(self.samples[0].split()) == 2

            for sample in self.samples:
                info = {'img_prefix': self.data_prefix}
                sample = sample.split()
                if self.has_labels:
                    info['img_info'] = {'filename': sample[0]}
                    info['gt_label'] = np.array(sample[1], dtype=np.int64)
                else:
                    info['img_info'] = {'filename': sample[0]}
                    info['gt_label'] = np.array(-1, dtype=np.int64)
                data_infos.append(info)
        return data_infos
