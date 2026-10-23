from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Explicitly load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from api.database import get_db_path
from api.routes import ask, stats, titles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Verify database and resources on application startup."""
    db_path = get_db_path()
    if not db_path.exists():
        logger.warning(
            "Database file not found at %s! Endpoints requiring database will fail. "
            "Please run 'python ingest.py' first.",
            db_path,
        )
    else:
        logger.info("Found database at %s. Ready to serve requests.", db_path)
    yield


app = FastAPI(
    title="Netflix Catalog Q&A API",
    description="REST API for structured catalogue search, statistics, and grounded RAG Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(titles.router)
app.include_router(stats.router)
app.include_router(ask.router)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Root entry point with documentation links."""
    return {
        "name": "Netflix Catalog Q&A API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health_check() -> dict[str, str]:
    """Service health verification endpoint."""
    db_path = get_db_path()
    return {
        "status": "healthy",
        "database": "connected" if db_path.exists() else "missing",
    }
