import pytest
from unittest.mock import MagicMock, patch
import numpy as np

@pytest.fixture(scope="session")
def mock_faiss_in_clutering():
    with patch("mmselfsup.utils.clustering.faiss") as faiss:
        
        yield faiss
#def mock_faiss():
#    mock = MagicMock()
#    mock.PCAMatrix().apply_py.return_value = np.random.rand(10, 8)
#    mock.GpuIndexFlatL2().search.return_value = (np.random.rand(1000, 6), np.random.rand(1000, 6))
#    yield mock