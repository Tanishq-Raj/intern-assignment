"""Data ingestion script for Netflix Catalog.

Reads data/netflix_titles.csv, cleans and normalises records,
and writes to a SQLite database at data/netflix.db.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = Path("data/netflix_titles.csv")
DB_PATH = Path("data/netflix.db")
DATE_INPUT_FORMAT = "%B %d, %Y"
DATE_ISO_FORMAT = "%Y-%m-%d"


def clean_string(val: object) -> str | None:
    """Strip whitespace and return None if empty or missing."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def parse_date_added(val: object) -> tuple[str | None, str | None]:
    """Parse date_added into (cleaned_raw_date, iso_date_str).

    Returns (None, None) if missing or unparseable.
    """
    raw_str = clean_string(val)
    if not raw_str:
        return None, None
    try:
        dt = datetime.strptime(raw_str, DATE_INPUT_FORMAT)
        return raw_str, dt.strftime(DATE_ISO_FORMAT)
    except ValueError:
        logger.warning("Failed to parse date_added value: '%s'", raw_str)
        return raw_str, None


def split_multi_values(val: object) -> list[str]:
    """Split comma-separated multi-value strings into cleaned items."""
    s = clean_string(val)
    if not s:
        return []
    items: list[str] = []
    for part in s.split(","):
        cleaned = part.strip()
        if cleaned:
            items.append(cleaned)
    return items


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean and normalise all columns in the Netflix titles dataframe."""
    stats = {
        "initial_rows": len(df),
        "missing_directors": int(df["director"].isna().sum()),
        "missing_cast": int(df["cast"].isna().sum()),
        "missing_country": int(df["country"].isna().sum()),
        "missing_date_added": int(df["date_added"].isna().sum()),
        "missing_rating": int(df["rating"].isna().sum()),
        "whitespace_trimmed_dates": int(
            df["date_added"].dropna().apply(lambda x: str(x) != str(x).strip()).sum()
        ),
        "whitespace_trimmed_titles": int(
            df["title"].dropna().apply(lambda x: str(x) != str(x).strip()).sum()
        ),
    }

    cleaned_df = df.copy()

    # Clean text columns
    text_columns = [
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    for col in text_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(clean_string)

    # Clean and standardize date_added
    date_tuples = cleaned_df["date_added"].apply(parse_date_added)
    cleaned_df["date_added"] = date_tuples.apply(lambda t: t[0])
    cleaned_df["date_added_iso"] = date_tuples.apply(lambda t: t[1])

    # Ensure show_id and release_year are correct types
    cleaned_df["show_id"] = cleaned_df["show_id"].astype(int)
    cleaned_df["release_year"] = cleaned_df["release_year"].astype(int)

    return cleaned_df, stats


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create database tables and indexes if they do not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS titles (
                show_id INTEGER PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                director TEXT,
                cast TEXT,
                country TEXT,
                date_added TEXT,
                date_added_iso TEXT,
                release_year INTEGER NOT NULL,
                rating TEXT,
                duration TEXT NOT NULL,
                listed_in TEXT NOT NULL,
                description TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_titles_type ON titles(type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_titles_release_year ON titles(release_year);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_titles_rating ON titles(rating);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS title_countries (
                show_id INTEGER NOT NULL,
                country TEXT NOT NULL,
                PRIMARY KEY (show_id, country),
                FOREIGN KEY (show_id) REFERENCES titles(show_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_title_countries_country ON title_countries(country);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS title_genres (
                show_id INTEGER NOT NULL,
                genre TEXT NOT NULL,
                PRIMARY KEY (show_id, genre),
                FOREIGN KEY (show_id) REFERENCES titles(show_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_title_genres_genre ON title_genres(genre);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS title_cast (
                show_id INTEGER NOT NULL,
                actor TEXT NOT NULL,
                PRIMARY KEY (show_id, actor),
                FOREIGN KEY (show_id) REFERENCES titles(show_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_title_cast_actor ON title_cast(actor);")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS title_directors (
                show_id INTEGER NOT NULL,
                director TEXT NOT NULL,
                PRIMARY KEY (show_id, director),
                FOREIGN KEY (show_id) REFERENCES titles(show_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_title_directors_dir ON title_directors(director);")

    return conn


def insert_titles(conn: sqlite3.Connection, df: pd.DataFrame) -> dict[str, int]:
    """Insert titles and normalized relations idempotently."""
    counts = {
        "titles_inserted": 0,
        "countries_inserted": 0,
        "genres_inserted": 0,
        "cast_inserted": 0,
        "directors_inserted": 0,
    }

    title_records = []
    country_records = []
    genre_records = []
    cast_records = []
    director_records = []

    for _, row in df.iterrows():
        sid = int(row["show_id"])
        title_records.append((
            sid,
            row["type"],
            row["title"],
            row["director"],
            row["cast"],
            row["country"],
            row["date_added"],
            row["date_added_iso"],
            int(row["release_year"]),
            row["rating"],
            row["duration"],
            row["listed_in"],
            row["description"],
        ))

        for c in set(split_multi_values(row["country"])):
            country_records.append((sid, c))

        for g in set(split_multi_values(row["listed_in"])):
            genre_records.append((sid, g))

        for a in set(split_multi_values(row["cast"])):
            cast_records.append((sid, a))

        for d in set(split_multi_values(row["director"])):
            director_records.append((sid, d))

    with conn:
        # Clear child tables first to ensure idempotent run
        conn.execute("DELETE FROM title_countries;")
        conn.execute("DELETE FROM title_genres;")
        conn.execute("DELETE FROM title_cast;")
        conn.execute("DELETE FROM title_directors;")

        conn.executemany("""
            INSERT INTO titles (
                show_id, type, title, director, cast, country,
                date_added, date_added_iso, release_year, rating,
                duration, listed_in, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(show_id) DO UPDATE SET
                type=excluded.type,
                title=excluded.title,
                director=excluded.director,
                cast=excluded.cast,
                country=excluded.country,
                date_added=excluded.date_added,
                date_added_iso=excluded.date_added_iso,
                release_year=excluded.release_year,
                rating=excluded.rating,
                duration=excluded.duration,
                listed_in=excluded.listed_in,
                description=excluded.description;
        """, title_records)
        counts["titles_inserted"] = len(title_records)

        conn.executemany(
            "INSERT OR IGNORE INTO title_countries (show_id, country) VALUES (?, ?);",
            country_records,
        )
        counts["countries_inserted"] = len(country_records)

        conn.executemany(
            "INSERT OR IGNORE INTO title_genres (show_id, genre) VALUES (?, ?);",
            genre_records,
        )
        counts["genres_inserted"] = len(genre_records)

        conn.executemany(
            "INSERT OR IGNORE INTO title_cast (show_id, actor) VALUES (?, ?);",
            cast_records,
        )
        counts["cast_inserted"] = len(cast_records)

        conn.executemany(
            "INSERT OR IGNORE INTO title_directors (show_id, director) VALUES (?, ?);",
            director_records,
        )
        counts["directors_inserted"] = len(director_records)

    return counts


def run_ingest(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> dict[str, int]:
    """Execute complete ingestion pipeline and return summary report statistics."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Source dataset not found at {csv_path}")

    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    logger.info("Cleaning data and parsing date attributes...")
    cleaned_df, cleaning_stats = clean_dataframe(df)

    logger.info("Initializing SQLite database at %s...", db_path)
    conn = init_db(db_path)

    try:
        logger.info("Inserting titles and normalized entities into database...")
        insert_stats = insert_titles(conn, cleaned_df)
    finally:
        conn.close()

    summary = {**cleaning_stats, **insert_stats}
    return summary


def print_summary(summary: dict[str, int]) -> None:
    """Print formatted ingestion summary audit report."""
    print("=" * 60)
    print("           NETFLIX CATALOG INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total rows read from CSV:       {summary.get('initial_rows', 0)}")
    print(f"Titles inserted / updated:      {summary.get('titles_inserted', 0)}")
    print(f"Normalized countries indexed:   {summary.get('countries_inserted', 0)}")
    print(f"Normalized genres indexed:      {summary.get('genres_inserted', 0)}")
    print(f"Normalized cast members:        {summary.get('cast_inserted', 0)}")
    print(f"Normalized directors:           {summary.get('directors_inserted', 0)}")
    print("-" * 60)
    print("DATA QUALITY & CLEANING AUDIT:")
    print(f"  Missing director values:      {summary.get('missing_directors', 0)}")
    print(f"  Missing cast values:          {summary.get('missing_cast', 0)}")
    print(f"  Missing country values:       {summary.get('missing_country', 0)}")
    print(f"  Missing date_added values:    {summary.get('missing_date_added', 0)}")
    print(f"  Missing rating values:        {summary.get('missing_rating', 0)}")
    print(f"  Whitespace trimmed in dates:  {summary.get('whitespace_trimmed_dates', 0)}")
    print(f"  Whitespace trimmed in titles: {summary.get('whitespace_trimmed_titles', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    audit_summary = run_ingest()
    print_summary(audit_summary)
