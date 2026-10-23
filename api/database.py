"""Database access layer for the Netflix Catalog API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sqlite3
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/netflix.db")


def get_db_path() -> Path:
    """Return the configured database path."""
    env_path = os.getenv("DATABASE_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def get_db_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection configured with row factory."""
    target_path = db_path or get_db_path()
    if not target_path.exists():
        raise FileNotFoundError(f"Database not found at {target_path}. Please run ingest.py.")
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


def query_titles(
    country: str | None = None,
    release_year: int | None = None,
    type_: str | None = None,
    rating: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db_path: Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Query titles with optional filters and deterministic pagination."""
    conn = get_db_connection(db_path)
    try:
        where_clauses: list[str] = []
        params: list[Any] = []

        if country:
            where_clauses.append(
                "show_id IN (SELECT show_id FROM title_countries WHERE country = ? COLLATE NOCASE)"
            )
            params.append(country.strip())

        if release_year is not None:
            where_clauses.append("release_year = ?")
            params.append(release_year)

        if type_:
            where_clauses.append("type = ? COLLATE NOCASE")
            params.append(type_.strip())

        if rating:
            where_clauses.append("rating = ? COLLATE NOCASE")
            params.append(rating.strip())

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total count query
        count_sql = f"SELECT COUNT(*) FROM titles{where_sql}"
        total = conn.execute(count_sql, params).fetchone()[0]

        # Data query with deterministic ordering
        data_sql = (
            f"SELECT * FROM titles{where_sql} "
            f"ORDER BY release_year DESC, show_id ASC LIMIT ? OFFSET ?"
        )
        data_params = params + [limit, offset]
        rows = conn.execute(data_sql, data_params).fetchall()
        titles = [dict(row) for row in rows]

        return titles, total
    finally:
        conn.close()


def get_title_by_id(show_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    """Retrieve a single title by its primary key identifier."""
    conn = get_db_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM titles WHERE show_id = ?", (show_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_stats_data(db_path: Path | None = None) -> dict[str, Any]:
    """Retrieve summary counts for total titles, titles by type, and top 10 countries."""
    conn = get_db_connection(db_path)
    try:
        # Total count
        total = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]

        # Count by type
        type_rows = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM titles GROUP BY type ORDER BY cnt DESC"
        ).fetchall()
        count_by_type = {row["type"]: row["cnt"] for row in type_rows}

        # Top 10 countries
        country_rows = conn.execute(
            """
            SELECT country, COUNT(DISTINCT show_id) as cnt
            FROM title_countries
            GROUP BY country
            ORDER BY cnt DESC, country ASC
            LIMIT 10
            """
        ).fetchall()
        top_countries = [{"country": row["country"], "count": row["cnt"]} for row in country_rows]

        return {
            "total_titles": total,
            "count_by_type": count_by_type,
            "count_by_country": top_countries,
        }
    finally:
        conn.close()
