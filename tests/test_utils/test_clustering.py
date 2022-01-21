import numpy as np
import pytest

from mmselfsup.utils.clustering import PIC, Kmeans


@pytest.fixture
def mock_faiss(mock_faiss_in_clutering):
    mock_faiss_in_clutering.PCAMatrix.return_value.apply_py.return_value = np.random.rand(10, 8)
    mock_faiss_in_clutering.GpuIndexFlatL2.return_value.search.return_value = (np.random.rand(1000, 6), np.random.rand(1000, 6))

@pytest.mark.parametrize("verbose", [True, False])
def test_kmeans(mock_faiss, verbose):
    fake_input = np.random.rand(10, 8).astype(np.float32)
    pca_dim = 2

    kmeans = Kmeans(2, pca_dim)
    loss = kmeans.cluster(fake_input, verbose=verbose)
    assert loss is not None

    with pytest.raises(AssertionError):
        loss = kmeans.cluster(np.random.rand(10, 8), verbose=verbose)


def test_pic(mock_faiss):
    fake_input = np.random.rand(1000, 16).astype(np.float32)
    pic = PIC(pca_dim=8)
    res = pic.cluster(fake_input)
    assert res == 0
