import math
from app.services.embeddings import embed_text


async def test_embed_returns_correct_dimension():
    result = await embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 384


async def test_embed_is_normalized():
    result = await embed_text("test input")
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-6


async def test_embed_empty_string():
    result = await embed_text("")
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)
