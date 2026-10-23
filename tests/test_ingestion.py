"""Tests for data ingestion and normalization pipeline."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd
from ingest import clean_dataframe, clean_string, init_db, insert_titles, parse_date_added


def test_clean_string() -> None:
    """Verify clean_string strips whitespace and converts empty values to None."""
    assert clean_string("  hello world  ") == "hello world"
    assert clean_string("") is None
    assert clean_string("   ") is None
    assert clean_string(None) is None


def test_parse_date_added() -> None:
    """Verify date_added parsing to ISO format and handling of malformed input."""
    raw, iso = parse_date_added(" September 9, 2019 ")
    assert raw == "September 9, 2019"
    assert iso == "2019-09-09"

    raw_invalid, iso_invalid = parse_date_added("Not a valid date")
    assert raw_invalid == "Not a valid date"
    assert iso_invalid is None

    assert parse_date_added(None) == (None, None)


def test_ingestion_cleaning_and_relations(tmp_path: Path, sample_csv: Path) -> None:
    """Verify dataframe cleaning, DB table population, and junction tables."""
    df = pd.read_csv(sample_csv)
    cleaned_df, stats = clean_dataframe(df)

    assert len(cleaned_df) == 5
    assert stats["whitespace_trimmed_dates"] >= 1
    # Check that date_added_iso was populated
    assert cleaned_df.loc[cleaned_df["show_id"] == 1001, "date_added_iso"].iloc[0] == "2019-09-09"

    db_path = tmp_path / "ingest_test.db"
    conn = init_db(db_path)
    try:
        counts = insert_titles(conn, cleaned_df)
        assert counts["titles_inserted"] == 5

        # Check multi-country entries (e.g., Roma has "Mexico, United States")
        cursor = conn.cursor()
        cursor.execute("SELECT country FROM title_countries WHERE show_id = 1005 ORDER BY country")
        countries = [r[0] for r in cursor.fetchall()]
        assert countries == ["Mexico", "United States"]

        # Check missing director for Stranger Things
        cursor.execute("SELECT director FROM titles WHERE show_id = 1003")
        assert cursor.fetchone()[0] is None
    finally:
        conn.close()


def test_ingestion_idempotency(tmp_path: Path, sample_csv: Path) -> None:
    """Verify that re-running ingestion does not create duplicate records."""
    df = pd.read_csv(sample_csv)
    cleaned_df, _ = clean_dataframe(df)

    db_path = tmp_path / "idempotent_test.db"
    conn = init_db(db_path)
    try:
        insert_titles(conn, cleaned_df)
        # Second run with exact same data
        insert_titles(conn, cleaned_df)

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM titles")
        assert cursor.fetchone()[0] == 5

        cursor.execute("SELECT COUNT(*) FROM title_countries")
        # Ensure junction table records were not multiplied
        assert cursor.fetchone()[0] == 6  # 1 each for 4 titles + 2 for Roma
    finally:
        conn.close()
