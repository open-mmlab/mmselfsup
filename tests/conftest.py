import pytest
from unittest.mock import patch


@pytest.fixture(scope="session")
def mock_faiss_in_clutering():
    with patch("mmselfsup.utils.clustering.faiss") as faiss:
        yield faiss