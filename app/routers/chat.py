import logging
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.config import get_settings
from app.models import User, JournalEntry
from app.schemas import ChatRequest, ChatResponse, ChatMessage
from app.auth import get_current_user
from app.services.embeddings import embed_text
from app.services.hybrid_retrieval import hybrid_search
from app.services.llm import get_llm, SYSTEM_PROMPT

logger = logging.getLogger("journal-ai")
router = APIRouter(prefix="/api/chat", tags=["chat"])

MAX_HISTORY_TURNS = 20


def _build_messages(body: ChatRequest, context: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in body.history[-MAX_HISTORY_TURNS * 2:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({
        "role": "user",
        "content": f"Context from journal entries:\n\n{context}\n\n---\n\nQuestion: {body.message}",
    })
    return messages


async def _get_rag_context(user_id: str, message: str, db: AsyncSession) -> tuple[list[dict], str]:
    query_embedding = await embed_text(message)
    results = await hybrid_search(user_id, query_embedding, message, top_k=5)

    context_parts = []
    sources = []
    for r in results:
        entry_result = await db.execute(
            select(JournalEntry).where(JournalEntry.id == r["entry_id"])
        )
        entry = entry_result.scalar_one_or_none()
        if entry is None:
            continue
        date_str = entry.created_at.strftime("%B %d, %Y")
        context_parts.append(f"[{date_str}] {r['text']}")
        sources.append({"entry_id": r["entry_id"], "date": date_str, "score": r["score"]})

    context = "\n\n".join(context_parts) if context_parts else "No relevant journal entries found."
    return sources, context


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    llm = get_llm()
    settings = get_settings()

    sources, context = await _get_rag_context(user.id, body.message, db)
    messages = _build_messages(body, context)

    logger.info("RAG query from user %s: %s", user.id, body.message)
    for s in sources:
        logger.info("  -> entry %s (%s) score=%.4f", s["entry_id"], s["date"], s["score"])

    logger.info("Sending %d sources to LLM (%s)", len(sources), settings.LLM_MODEL)
    answer = await llm.chat_completion(messages)
    logger.info("LLM response received (%d chars)", len(answer))
    return ChatResponse(answer=answer, sources=sources)


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    llm = get_llm()
    settings = get_settings()

    sources, context = await _get_rag_context(user.id, body.message, db)
    messages = _build_messages(body, context)

    logger.info("Streaming RAG query from user %s: %s", user.id, body.message)
    for s in sources:
        logger.info("  -> entry %s (%s) score=%.4f", s["entry_id"], s["date"], s["score"])

    logger.info("Streaming from LLM (%s) with %d sources", settings.LLM_MODEL, len(sources))

    async def event_generator():
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        async for chunk in llm.chat_completion_stream(messages):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
