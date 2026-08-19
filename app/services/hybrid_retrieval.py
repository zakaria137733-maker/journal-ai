import asyncio
import logging
import re
from rank_bm25 import BM25Okapi

from app.services.vector_store import search_similar, fetch_user_points

logger = logging.getLogger("journal-ai")

_bm25_cache: dict[str, BM25Okapi] = {}
_entry_map_cache: dict[str, dict[str, str]] = {}

RRF_K = 10


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


async def _get_bm25_index(user_id: str) -> tuple[BM25Okapi, dict[str, str]]:
    if user_id in _bm25_cache:
        return _bm25_cache[user_id], _entry_map_cache[user_id]

    points = await fetch_user_points(user_id)
    entry_map = {p["entry_id"]: p["text"] for p in points}
    corpus = [_tokenize(p["text"]) for p in points]

    if corpus:
        bm25 = BM25Okapi(corpus)
    else:
        bm25 = BM25Okapi([[]])

    _bm25_cache[user_id] = bm25
    _entry_map_cache[user_id] = entry_map
    return bm25, entry_map


def invalidate_bm25_cache(user_id: str):
    _bm25_cache.pop(user_id, None)
    _entry_map_cache.pop(user_id, None)


async def hybrid_search(user_id: str, query_embedding: list[float], query_text: str, top_k: int = 5) -> list[dict]:
    vector_task = search_similar(user_id, query_embedding, top_k=top_k * 2)

    bm25, entry_map = await _get_bm25_index(user_id)

    bm25_results = []
    if entry_map:
        tokenized_query = _tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in ranked_indices[: top_k * 2]:
            if scores[idx] > 0:
                entry_id = list(entry_map.keys())[idx]
                bm25_results.append({"entry_id": entry_id, "text": entry_map[entry_id], "score": float(scores[idx])})

    vector_results = await vector_task

    logger.info("Hybrid search: %d vector + %d BM25 candidates for user %s", len(vector_results), len(bm25_results), user_id)
    fused = _rrf_fuse(vector_results, bm25_results, top_k)
    logger.info("RRF fusion: %d final results", len(fused))
    return fused


def _rrf_fuse(vector_results: list[dict], bm25_results: list[dict], top_k: int) -> list[dict]:
    scores: dict[str, float] = {}
    entries: dict[str, dict] = {}

    for rank, r in enumerate(vector_results):
        eid = r["entry_id"]
        scores[eid] = scores.get(eid, 0) + 1.0 / (RRF_K + rank + 1)
        entries[eid] = r

    for rank, r in enumerate(bm25_results):
        eid = r["entry_id"]
        scores[eid] = scores.get(eid, 0) + 1.0 / (RRF_K + rank + 1)
        if eid not in entries:
            entries[eid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for eid, score in ranked[:top_k]:
        entry = entries[eid]
        results.append({"entry_id": eid, "text": entry["text"], "score": score})

    return results
