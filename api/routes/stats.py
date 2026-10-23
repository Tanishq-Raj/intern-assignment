"""Statistics route handlers for catalogue summary metrics."""

from __future__ import annotations

from fastapi import APIRouter
from api.database import get_stats_data
from api.models import CountryStat, StatsResponse

router = APIRouter(tags=["Statistics"])


@router.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Return catalogue summary stats: total titles, breakdown by type, top 10 countries."""
    raw_stats = get_stats_data()
    return StatsResponse(
        total_titles=raw_stats["total_titles"],
        count_by_type=raw_stats["count_by_type"],
        count_by_country=[CountryStat(**c) for c in raw_stats["count_by_country"]],
    )
