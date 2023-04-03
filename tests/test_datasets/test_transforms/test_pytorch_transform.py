# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from PIL import Image

from mmselfsup.datasets.transforms import MAERandomResizedCrop


def test_mae_random_resized_crop():
    transform = dict(size=224, interpolation=3)

    img_numpy = torch.rand((512, 512, 3)).numpy() * 255
    img_numpy = img_numpy.astype(np.uint8)
    img_pil = Image.fromarray(img_numpy)
    results = {'img': img_pil}
    module = MAERandomResizedCrop(**transform)

    results = module(results)
    results['img'] = np.array(results['img'])

    # test transform
    assert list(results['img'].shape) == [224, 224, 3]

    # test repr
    assert isinstance(str(module), str)
