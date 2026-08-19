import asyncio
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
)
from app.config import get_settings

logger = logging.getLogger("journal-ai")

_client = None
COLLECTION_initialized = False


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return _client


def _ensure_collection_sync():
    global COLLECTION_initialized
    if COLLECTION_initialized:
        return
    client = get_client()
    settings = get_settings()
    collections = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in collections:
        logger.info("Creating collection: %s", settings.QDRANT_COLLECTION)
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="user_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Payload index on 'user_id' ensured.")
    COLLECTION_initialized = True


async def ensure_collection():
    await asyncio.to_thread(_ensure_collection_sync)


def _upsert_entry_sync(entry_id: str, user_id: str, text: str, embedding: list[float]):
    client = get_client()
    settings = get_settings()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_id))
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"entry_id": entry_id, "user_id": user_id, "text": text},
            )
        ],
    )
    logger.info("Upserted point %s (entry %s) to Qdrant", point_id, entry_id)


async def upsert_entry(entry_id: str, user_id: str, text: str, embedding: list[float]):
    await asyncio.to_thread(_upsert_entry_sync, entry_id, user_id, text, embedding)


def _search_similar_sync(user_id: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    client = get_client()
    settings = get_settings()
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=top_k,
        with_payload=True,
    )
    results = [
        {"entry_id": r.payload["entry_id"], "text": r.payload["text"], "score": r.score}
        for r in results.points
    ]
    logger.info("Vector search: %d results for user %s (top_k=%d)", len(results), user_id, top_k)
    return results


async def search_similar(user_id: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    return await asyncio.to_thread(_search_similar_sync, user_id, query_embedding, top_k)


def _fetch_user_points_sync(user_id: str) -> list[dict]:
    client = get_client()
    settings = get_settings()
    all_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, offset = result
        for p in points:
            all_points.append({"entry_id": p.payload["entry_id"], "text": p.payload["text"]})
        if offset is None:
            break
    return all_points


async def fetch_user_points(user_id: str) -> list[dict]:
    return await asyncio.to_thread(_fetch_user_points_sync, user_id)


def _delete_entry_sync(entry_id: str):
    client = get_client()
    settings = get_settings()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_id))
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=PointIdsList(points=[point_id]),
    )
    logger.info("Deleted point %s (entry %s) from Qdrant", point_id, entry_id)


async def delete_entry(entry_id: str):
    await asyncio.to_thread(_delete_entry_sync, entry_id)
