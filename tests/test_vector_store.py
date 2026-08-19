import uuid
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from app.services.vector_store import (
    upsert_entry,
    search_similar,
    delete_entry,
    ensure_collection,
)


async def test_upsert_calls_client(mock_qdrant):
    entry_id = str(uuid.uuid4())
    user_id = "user-1"
    text = "Today was great"
    embedding = [0.1] * 384

    await upsert_entry(entry_id, user_id, text, embedding)

    mock_qdrant.upsert.assert_called_once()
    call = mock_qdrant.upsert.call_args
    assert call.kwargs["collection_name"] == "test_collection"
    point = call.kwargs["points"][0]
    assert isinstance(point, PointStruct)
    assert point.vector == embedding
    assert point.payload["entry_id"] == entry_id
    assert point.payload["user_id"] == user_id
    assert point.payload["text"] == text
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_id))
    assert str(point.id) == expected_id


async def test_search_filters_by_user_id(mock_qdrant):
    user_id = "user-A"
    query_embedding = [0.2] * 384

    await search_similar(user_id, query_embedding, top_k=5)

    mock_qdrant.query_points.assert_called_once()
    call = mock_qdrant.query_points.call_args
    assert call.kwargs["collection_name"] == "test_collection"
    query_filter = call.kwargs["query_filter"]
    assert isinstance(query_filter, Filter)
    condition = query_filter.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "user_id"
    assert isinstance(condition.match, MatchValue)
    assert condition.match.value == user_id


async def test_delete_removes_point(mock_qdrant):
    entry_id = str(uuid.uuid4())
    expected_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_id))

    await delete_entry(entry_id)

    mock_qdrant.delete.assert_called_once()
    call = mock_qdrant.delete.call_args
    assert call.kwargs["collection_name"] == "test_collection"
    selector = call.kwargs["points_selector"]
    assert expected_point_id in selector.points


async def test_ensure_collection_creates_if_missing(mock_qdrant):
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mock_qdrant.create_collection = MagicMock()
    mock_qdrant.create_payload_index = MagicMock()

    import app.services.vector_store as vs
    vs.COLLECTION_initialized = False

    await ensure_collection()

    mock_qdrant.create_collection.assert_called_once()
    create_call = mock_qdrant.create_collection.call_args
    assert create_call.kwargs["collection_name"] == "test_collection"
    vec_params = create_call.kwargs["vectors_config"]
    assert isinstance(vec_params, VectorParams)
    assert vec_params.size == 384
    assert vec_params.distance == Distance.COSINE

    mock_qdrant.create_payload_index.assert_called_once()
    idx_call = mock_qdrant.create_payload_index.call_args
    assert idx_call.kwargs["field_name"] == "user_id"
    assert idx_call.kwargs["field_schema"] == PayloadSchemaType.KEYWORD
