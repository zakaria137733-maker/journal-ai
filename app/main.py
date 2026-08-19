import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.database import init_db
from app.services.vector_store import ensure_collection
from app.routers import auth, entries, chat


def setup_logging():
    handler = RotatingFileHandler("server.log", maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger("journal-ai")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(stream)


setup_logging()
logger = logging.getLogger("journal-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing database...")
    await init_db()
    logger.info("Initializing Qdrant collection...")
    await ensure_collection()
    logger.info("Ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(title="Journal AI", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(entries.router)
app.include_router(chat.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")