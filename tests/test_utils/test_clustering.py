import numpy as np
import pytest
import torch

from mmselfsup.utils.clustering import PIC, Kmeans


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is not available.')
def test_kmeans():
    fake_input = np.random.rand(10, 8).astype(np.float32)
    pca_dim = 2

    kmeans = Kmeans(2, pca_dim)
    loss = kmeans.cluster(fake_input)
    assert loss is not None

    with pytest.raises(AssertionError):
        loss = kmeans.cluster(np.random.rand(10, 8))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is not available.')
def test_pic():
    fake_input = np.random.rand(1000, 16).astype(np.float32)
    pic = PIC(pca_dim=8)
    res = pic.cluster(fake_input)
    assert res == 0
