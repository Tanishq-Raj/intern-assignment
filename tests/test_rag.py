"""Tests for RAG pipeline: dense retrieval, source mapping, evidence policies, and LLM mocking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest
from rag.generator import INSUFFICIENT_EVIDENCE_ANSWER, generate_answer
from rag.retriever import reset_index_cache, retrieve_titles


def test_retrieval_finds_relevant_titles(test_index: Path) -> None:
    """Verify dense retrieval returns relevant titles with scores matching search query."""
    reset_index_cache()
    results = retrieve_titles(
        query="dreams and stealing secrets", top_k=3, index_path=test_index
    )
    assert len(results) > 0
    top_title = results[0]
    assert top_title["title"] == "Inception"
    assert top_title["show_id"] == 1002
    assert top_title["score"] >= 0.25


def test_retrieval_insufficient_evidence_for_gibberish(test_index: Path) -> None:
    """Verify irrelevant query returns no results when score threshold is not met."""
    reset_index_cache()
    results = retrieve_titles(
        query="xyzqwerty unmatched query 987654321",
        min_score=0.25,
        index_path=test_index,
    )
    assert results == []


def test_generator_insufficient_evidence() -> None:
    """Verify generator returns standard insufficient evidence message when given empty context."""
    answer, ids = generate_answer("What is the plot of an unlisted movie?", [])
    assert answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert ids == []


def test_generator_offline_grounded_answer() -> None:
    """Verify offline generator cites real show_ids and titles without external APIs."""
    sample_items = [
        {
            "show_id": 1004,
            "title": "3 Idiots",
            "type": "Movie",
            "release_year": 2009,
            "country": "India",
            "listed_in": "Comedies",
            "director": "Rajkumar Hirani",
            "cast": "Aamir Khan",
            "description": "Two friends search for their companion.",
            "score": 0.85,
        }
    ]
    answer, ids = generate_answer("Recommend an Indian comedy", sample_items)
    assert ids == [1004]
    assert "3 Idiots" in answer
    assert "1004" in answer


def test_generator_subjective_attribute_policy() -> None:
    """Verify generator honestly discloses lack of evidence for subjective claims like 'strong female lead'."""
    sample_items = [
        {
            "show_id": 1004,
            "title": "3 Idiots",
            "type": "Movie",
            "release_year": 2009,
            "country": "India",
            "listed_in": "Comedies",
            "director": "Rajkumar Hirani",
            "cast": "Aamir Khan",
            "description": "Two friends search for their companion.",
            "score": 0.65,
        }
    ]
    answer, ids = generate_answer(
        "Suggest me a comedy movie from India with a strong female lead", sample_items
    )
    assert ids == [1004]
    # Invariant: Must disclose that the catalogue does not track/prove subjective attribute
    assert "strong female lead" in answer.lower()
    assert "subjective" in answer.lower() or "does not track" in answer.lower() or "evidence" in answer.lower()


def test_generator_mocked_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify external LLM provider integration with mocked API response."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-mock-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    sample_items = [
        {
            "show_id": 1002,
            "title": "Inception",
            "type": "Movie",
            "release_year": 2010,
            "country": "United States",
            "listed_in": "Action",
            "director": "Christopher Nolan",
            "cast": "Leonardo DiCaprio",
            "description": "A thief enters dreams.",
            "score": 0.9,
        }
    ]

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Inception (show_id: 1002) is a sci-fi thriller about entering dreams."
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        answer, ids = generate_answer("What is Inception about?", sample_items)
        assert ids == [1002]
        assert "Inception" in answer


def test_generator_mocked_llm_failure_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify graceful fallback to offline generator when external LLM call raises an exception."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-mock-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    sample_items = [
        {
            "show_id": 1001,
            "title": "Delhi Safari",
            "type": "Movie",
            "release_year": 2012,
            "country": "India",
            "listed_in": "Comedies",
            "director": "Nikhil Advani",
            "cast": "Govinda",
            "description": "Animals protect their habitat.",
            "score": 0.7,
        }
    ]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API rate limit exceeded")
        mock_openai_cls.return_value = mock_client

        answer, ids = generate_answer("Tell me about Delhi Safari", sample_items)
        assert ids == [1001]
        assert "Delhi Safari" in answer
        assert "(Note: Generated via offline grounded synthesizer." in answer


def test_ask_endpoint_integration(client: TestClient) -> None:
    """Verify /ask HTTP endpoint returns question, answer, and sources invariant."""
    payload = {"question": "Suggest a movie about entering dreams"}
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == payload["question"]
    assert len(data["answer"]) > 0

    # Invariant: If an answer is provided, show_ids and sources must match and come from retrieved records
    if data["show_ids"]:
        assert len(data["sources"]) == len(data["show_ids"])
        source_ids = [s["show_id"] for s in data["sources"]]
        assert set(data["show_ids"]) == set(source_ids)
