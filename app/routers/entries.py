import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User, JournalEntry
from app.schemas import EntryCreate, EntryUpdate, EntryResponse, EntryListResponse
from app.auth import get_current_user
from app.services.embeddings import embed_text
from app.services.vector_store import upsert_entry, delete_entry
from app.services.hybrid_retrieval import invalidate_bm25_cache

logger = logging.getLogger("journal-ai")
router = APIRouter(prefix="/api/entries", tags=["entries"])


@router.get("", response_model=EntryListResponse)
async def list_entries(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    search: str = Query(None, min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    query = select(JournalEntry).where(JournalEntry.user_id == user.id)
    count_query = select(func.count()).select_from(JournalEntry).where(JournalEntry.user_id == user.id)

    if search:
        like_pattern = f"%{search}%"
        query = query.where(JournalEntry.content.ilike(like_pattern))
        count_query = count_query.where(JournalEntry.content.ilike(like_pattern))

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    query = query.order_by(JournalEntry.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    entries = result.scalars().all()

    return EntryListResponse(entries=entries, total=total)


@router.post("", response_model=EntryResponse, status_code=201)
async def create_entry(
    body: EntryCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    entry = JournalEntry(user_id=user.id, content=body.content)
    db.add(entry)
    await db.commit()
    await db.refresh(entry)

    embedding = await embed_text(body.content)
    await upsert_entry(entry.id, user.id, body.content, embedding)
    invalidate_bm25_cache(user.id)
    logger.info("Created entry %s — embedded + synced to Qdrant", entry.id)

    return entry


@router.get("/{entry_id}", response_model=EntryResponse)
async def get_entry(
    entry_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JournalEntry).where(
            JournalEntry.id == entry_id, JournalEntry.user_id == user.id
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.put("/{entry_id}", response_model=EntryResponse)
async def update_entry(
    entry_id: str,
    body: EntryUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JournalEntry).where(
            JournalEntry.id == entry_id, JournalEntry.user_id == user.id
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    entry.content = body.content
    await db.commit()
    await db.refresh(entry)

    embedding = await embed_text(body.content)
    await upsert_entry(entry.id, user.id, body.content, embedding)
    invalidate_bm25_cache(user.id)
    logger.info("Updated entry %s — re-embedded + synced to Qdrant", entry.id)

    return entry


@router.delete("/{entry_id}", status_code=204)
async def delete_entry_endpoint(
    entry_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JournalEntry).where(
            JournalEntry.id == entry_id, JournalEntry.user_id == user.id
        )
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    await db.delete(entry)
    await db.commit()

    await delete_entry(entry_id)
    invalidate_bm25_cache(user.id)
    logger.info("Deleted entry %s — removed from SQLite + Qdrant", entry_id)
