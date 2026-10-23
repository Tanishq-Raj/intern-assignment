"""Catalogue indexer module for RAG pipeline.

Uses local SentenceTransformer ('all-MiniLM-L6-v2') to compute dense semantic embeddings
and persists the index package for fast cosine-similarity retrieval.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
import sqlite3
from typing import Any
import numpy as np
import pandas as pd

# Prevent TensorFlow import issues on Windows
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/netflix.db")
DEFAULT_INDEX_PATH = Path("data/rag_index.pkl")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

_MODEL_INSTANCE: Any = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Return a singleton instance of the local SentenceTransformer model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model '%s'...", model_name)
        _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


def reset_model_instance() -> None:
    """Reset cached embedding model instance (useful for test isolation)."""
    global _MODEL_INSTANCE
    _MODEL_INSTANCE = None


def create_document_representation(row: pd.Series | dict[str, Any]) -> str:
    """Format catalogue item metadata into a unified searchable text representation."""
    rating_val = row.get("rating") or "Unrated"
    country_val = row.get("country") or "Unknown"
    director_val = row.get("director") or "Unknown"
    cast_val = row.get("cast") or "Unknown"

    parts = [
        f"Title: {row.get('title', '')}",
        f"Type: {row.get('type', '')}",
        f"Release Year: {row.get('release_year', '')}",
        f"Rating: {rating_val}",
        f"Country: {country_val}",
        f"Genres: {row.get('listed_in', '')}",
        f"Director: {director_val}",
        f"Cast: {cast_val}",
        f"Description: {row.get('description', '')}",
    ]
    return " | ".join(parts)


def fetch_catalog_dataframe(db_path: Path) -> pd.DataFrame:
    """Load all titles from the SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at {db_path}. Run ingest.py first.")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM titles ORDER BY show_id ASC", conn)
        return df
    finally:
        conn.close()


def build_index_data(
    df: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME
) -> dict[str, Any]:
    """Encode catalogue documents into normalized dense embeddings and package metadata."""
    logger.info("Building document representations for %d titles...", len(df))
    documents = df.apply(create_document_representation, axis=1).tolist()

    model = get_embedding_model(model_name)
    logger.info("Computing dense embeddings with %s (batch_size=%d)...", model_name, BATCH_SIZE)
    embeddings = model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    metadata_records = df[
        [
            "show_id",
            "type",
            "title",
            "director",
            "cast",
            "country",
            "release_year",
            "rating",
            "duration",
            "listed_in",
            "description",
        ]
    ].to_dict(orient="records")

    return {
        "model_name": model_name,
        "embeddings": embeddings,
        "show_ids": df["show_id"].to_numpy(dtype=np.int64),
        "metadata": metadata_records,
        "total_docs": len(metadata_records),
    }


def save_index(index_data: dict[str, Any], index_path: Path) -> None:
    """Save index package to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "RAG dense index saved to %s (%d documents, embedding dim: %d)",
        index_path,
        index_data["total_docs"],
        index_data["embeddings"].shape[1],
    )


def load_index(index_path: Path) -> dict[str, Any]:
    """Load cached index package from disk."""
    if not index_path.exists():
        raise FileNotFoundError(f"RAG index not found at {index_path}")
    with open(index_path, "rb") as f:
        return pickle.load(f)


def build_and_save_index(
    db_path: Path = DEFAULT_DB_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Orchestrate end-to-end dense index construction and persistence."""
    df = fetch_catalog_dataframe(db_path)
    index_data = build_index_data(df, model_name=model_name)
    save_index(index_data, index_path)
    return index_data


if __name__ == "__main__":
    build_and_save_index()
