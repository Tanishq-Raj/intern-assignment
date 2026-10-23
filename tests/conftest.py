"""Pytest fixtures for isolated testing of ingestion, API, and RAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient
import pandas as pd
import pytest
from api.database import get_db_path
from api.main import app
from ingest import clean_dataframe, init_db, insert_titles
from rag.indexer import build_and_save_index
from rag.retriever import reset_index_cache

SAMPLE_CSV_DATA = '''show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description
1001,Movie,Delhi Safari,Nikhil Advani,"Govinda, Akshaye Khanna",India," September 9, 2019 ",2012,PG,93 min,"Children & Family Movies, Comedies","A group of animals journey to Delhi to protect their habitat."
1002,Movie,Inception,Christopher Nolan,"Leonardo DiCaprio, Joseph Gordon-Levitt",United States,"July 16, 2010",2010,PG-13,148 min,"Action & Adventure, Sci-Fi & Fantasy","A thief enters people's dreams to steal secrets."
1003,TV Show,Stranger Things,,,"United States","July 15, 2016",2016,TV-14,3 Seasons,"Sci-Fi & Fantasy, TV Dramas","A young boy vanishes and a secret experiment unfolds."
1004,Movie,3 Idiots,Rajkumar Hirani,"Aamir Khan, Kareena Kapoor",India,"August 1, 2019",2009,PG-13,170 min,"Comedies, Dramas, International Movies","Two friends search for their long-lost companion from engineering college."
1005,Movie,Roma,Alfonso Cuarón,"Yalitza Aparicio, Marina de Tavira","Mexico, United States","December 14, 2018",2018,R,135 min,"Dramas, Independent Movies, International Movies","A vivid emotional portrait of a domestic worker in 1970s Mexico."
'''


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file populated with sample catalog records."""
    csv_file = tmp_path / "test_titles.csv"
    csv_file.write_text(SAMPLE_CSV_DATA.strip(), encoding="utf-8")
    return csv_file


@pytest.fixture
def test_db(tmp_path: Path, sample_csv: Path) -> Path:
    """Initialize an isolated test SQLite database containing sample titles."""
    db_file = tmp_path / "test_netflix.db"
    df = pd.read_csv(sample_csv)
    cleaned_df, _ = clean_dataframe(df)

    conn = init_db(db_file)
    try:
        insert_titles(conn, cleaned_df)
    finally:
        conn.close()

    return db_file


@pytest.fixture
def test_index(tmp_path: Path, test_db: Path) -> Path:
    """Build and save an isolated test RAG index for sample data."""
    index_file = tmp_path / "test_rag_index.pkl"
    build_and_save_index(db_path=test_db, index_path=index_file)
    return index_file


@pytest.fixture
def client(test_db: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient configured with isolated test database."""
    monkeypatch.setenv("DATABASE_PATH", str(test_db))
    with TestClient(app) as test_client:
        yield test_client
