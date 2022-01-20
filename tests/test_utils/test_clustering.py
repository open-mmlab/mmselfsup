import numpy as np
import pytest
import torch

from mmselfsup.utils.clustering import PIC, Kmeans


def test_kmeans():
    fake_input = np.random.rand(10, 8).astype(np.float32)
    pca_dim = 2

    kmeans = Kmeans(2, pca_dim)
    for verbose in [True, False]:
        loss = kmeans.cluster(fake_input, verbose=verbose)
        assert loss is not None

        with pytest.raises(AssertionError):
            loss = kmeans.cluster(np.random.rand(10, 8), verbose=verbose)


def test_pic():
    fake_input = np.random.rand(1000, 16).astype(np.float32)
    pic = PIC(pca_dim=8)
    res = pic.cluster(fake_input)
    assert res == 0
