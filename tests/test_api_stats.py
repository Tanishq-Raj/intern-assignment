"""Tests for /stats endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_stats_aggregates(client: TestClient) -> None:
    """Verify stats endpoint returns accurate totals, type counts, and top countries."""
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_titles"] == 5
    assert data["count_by_type"]["Movie"] == 4
    assert data["count_by_type"]["TV Show"] == 1

    countries = {c["country"]: c["count"] for c in data["count_by_country"]}
    # In sample: Inception, Stranger Things, Roma are US -> 3
    assert countries["United States"] == 3
    # Delhi Safari, 3 Idiots are India -> 2
    assert countries["India"] == 2
    # Roma is Mexico -> 1
    assert countries["Mexico"] == 1
