import asyncio
import logging
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


def _embed_sync(text: str) -> list[float]:
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


async def embed_text(text: str) -> list[float]:
    logger.info("Embedding text (%d chars): %.80s...", len(text), text)
    result = await asyncio.to_thread(_embed_sync, text)
    logger.info("Embedding complete — vector dim=%d", len(result))
    return result
