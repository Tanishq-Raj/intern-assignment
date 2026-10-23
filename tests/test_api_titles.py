"""Tests for /titles and /titles/{show_id} endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_titles_no_filter(client: TestClient) -> None:
    """Verify listing all titles without filter returns total count and results."""
    resp = client.get("/titles")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["titles"]) == 5
    assert data["page"] == 1


def test_list_titles_filter_country(client: TestClient) -> None:
    """Verify filtering by country returns only matching titles (including co-productions)."""
    resp = client.get("/titles?country=India")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    show_ids = [t["show_id"] for t in data["titles"]]
    assert set(show_ids) == {1001, 1004}

    # Verify co-production match for Mexico
    resp_mexico = client.get("/titles?country=Mexico")
    assert resp_mexico.status_code == 200
    mexico_ids = [t["show_id"] for t in resp_mexico.json()["titles"]]
    assert 1005 in mexico_ids


def test_list_titles_filter_type_and_year(client: TestClient) -> None:
    """Verify compound filtering by type and release year."""
    resp = client.get("/titles?type=TV Show&release_year=2016")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["titles"][0]["title"] == "Stranger Things"


def test_list_titles_pagination(client: TestClient) -> None:
    """Verify pagination limits page size and navigates pages deterministically."""
    resp_p1 = client.get("/titles?page=1&page_size=2")
    assert resp_p1.status_code == 200
    data_p1 = resp_p1.json()
    assert data_p1["page"] == 1
    assert data_p1["page_size"] == 2
    assert data_p1["total"] == 5
    assert data_p1["total_pages"] == 3
    assert len(data_p1["titles"]) == 2

    resp_p2 = client.get("/titles?page=2&page_size=2")
    assert resp_p2.status_code == 200
    data_p2 = resp_p2.json()
    assert len(data_p2["titles"]) == 2

    # Disjoint check
    p1_ids = {t["show_id"] for t in data_p1["titles"]}
    p2_ids = {t["show_id"] for t in data_p2["titles"]}
    assert p1_ids.isdisjoint(p2_ids)


def test_list_titles_validation_errors(client: TestClient) -> None:
    """Verify invalid pagination query params return 422 HTTP validation errors."""
    resp_bad_page = client.get("/titles?page=0")
    assert resp_bad_page.status_code == 422

    resp_bad_size = client.get("/titles?page_size=500")
    assert resp_bad_size.status_code == 422


def test_get_title_by_id_found(client: TestClient) -> None:
    """Verify fetching an existing title returns correct details."""
    resp = client.get("/titles/1002")
    assert resp.status_code == 200
    data = resp.json()
    assert data["show_id"] == 1002
    assert data["title"] == "Inception"
    assert data["director"] == "Christopher Nolan"


def test_get_title_by_id_not_found(client: TestClient) -> None:
    """Verify requesting a non-existent show_id returns 404 with detail message."""
    resp = client.get("/titles/999999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
