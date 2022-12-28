# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import mmengine
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mmselfsup.registry import DATASETS


def selective_search(image, method='fast'):
    # initialize OpenCV's selective search implementation
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set the input image
    ss.setBaseImage(image)
    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if method == 'fast':
        # print("[INFO] using *fast* selective search")
        ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
        # print("[INFO] using *quality* selective search")
        ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    boxes = ss.process()
    return boxes


def box_filter(boxes, min_size=None, max_ratio=None, topN=None):
    proposal = []

    for box in boxes:
        # Calculate width and height of the box
        w, h = box[2], box[3]

        # Filter for size
        if min_size:
            if w < min_size or h < min_size:
                continue

        # Filter for box ratio
        if max_ratio:
            if w / h > max_ratio or h / w > max_ratio:
                continue

        proposal.append(box)

    if topN:
        if topN <= len(proposal):
            return proposal[:topN]
        else:
            return proposal
    else:
        return proposal


@DATASETS.register_module()
class SSDataset(Dataset):
    """Dataset for generating selective search proposals."""

    def __init__(self,
                 root,
                 json_file,
                 memcached=False,
                 mclient_path=None,
                 method='fast',
                 min_size=None,
                 max_ratio=None,
                 topN=None):
        data = mmengine.load(json_file)
        self.fns = [item['file_name'] for item in data['images']]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.method = method
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.topN = topN

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        # if self.memcached:
        #     self._init_memcached()
        # if self.memcached:
        #     img = self.mc_loader(self.fns[idx])
        # else:
        #     img = Image.open(self.fns[idx])
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = selective_search(img_cv2, self.method)
        if self.topN is not None:
            boxes = box_filter(boxes, self.min_size, self.max_ratio, self.topN)
        boxes = torch.from_numpy(np.array(boxes))
        # bbox: Bx4
        # B is the total number of original/topN selective search bboxes
        return dict(bbox=boxes)
