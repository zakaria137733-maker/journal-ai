import asyncio
import logging
import re
from sentence_transformers import SentenceTransformer
from app.config import get_settings

logger = logging.getLogger("journal-ai")

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def chunk_text(text: str, max_chars: int = 500, overlap: int = 100) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sentence
        else:
            current = current + " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def _embed_sync(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


async def embed_text(text: str) -> list[float]:
    logger.info("Embedding text (%d chars): %.80s...", len(text), text)
    result = await asyncio.to_thread(_embed_sync, [text])
    logger.info("Embedding complete — vector dim=%d", len(result[0]))
    return result[0]


async def embed_all(texts: list[str]) -> list[list[float]]:
    logger.info("Embedding %d chunks", len(texts))
    results = await asyncio.to_thread(_embed_sync, texts)
    logger.info("Embedded %d chunks — vector dim=%d", len(results), len(results[0]))
    return results
