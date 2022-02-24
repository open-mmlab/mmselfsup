# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import random
import string
import tempfile

import numpy as np
import pytest
from PIL import Image

from mmselfsup.datasets.utils import check_integrity, rm_suffix, to_numpy


def test_to_numpy():
    pil_img = Image.open(
        osp.join(osp.dirname(__file__), '..', 'data', 'color.jpg'))
    np_img = to_numpy(pil_img)
    assert type(np_img) == np.ndarray
    if np_img.ndim < 3:
        assert np_img.shape[0] == 1
    elif np_img.ndim == 3:
        assert np_img.shape[0] == 3


@pytest.mark.skipif(
    platform.system() == 'Windows', reason='Windows permission')
def test_dataset_utils():
    # test rm_suffix
    assert rm_suffix('a.jpg') == 'a'
    assert rm_suffix('a.bak.jpg') == 'a.bak'
    assert rm_suffix('a.bak.jpg', suffix='.jpg') == 'a.bak'
    assert rm_suffix('a.bak.jpg', suffix='.bak.jpg') == 'a'

    # test check_integrity
    rand_file = ''.join(random.sample(string.ascii_letters, 10))
    assert not check_integrity(rand_file, md5=None)
    assert not check_integrity(rand_file, md5=2333)
    tmp_file = tempfile.NamedTemporaryFile()
    assert check_integrity(tmp_file.name, md5=None)
    assert not check_integrity(tmp_file.name, md5=2333)
