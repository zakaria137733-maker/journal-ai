import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.query_points.return_value = MagicMock(points=[])
    client.scroll.return_value = ([], None)
    return client


@pytest.fixture(autouse=True)
def patch_vector_store(mock_qdrant):
    import app.services.vector_store as vs
    vs._client = mock_qdrant
    vs.COLLECTION_initialized = False
    yield
    vs._client = None
    vs.COLLECTION_initialized = False


@pytest.fixture(autouse=True)
def patch_settings():
    mock_settings = MagicMock()
    mock_settings.SECRET_KEY = "test-secret-key"
    mock_settings.QDRANT_URL = "http://localhost:6333"
    mock_settings.QDRANT_API_KEY = "test-key"
    mock_settings.QDRANT_COLLECTION = "test_collection"
    mock_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    with patch("app.config.get_settings", return_value=mock_settings):
        with patch("app.services.vector_store.get_settings", return_value=mock_settings):
            yield
