"""Titles route handlers for listing and retrieving Netflix titles."""

from __future__ import annotations

import math
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from api.database import get_title_by_id, query_titles
from api.models import TitleListResponse, TitleResponse

router = APIRouter(tags=["Titles"])

MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20


@router.get("/titles", response_model=TitleListResponse)
def list_titles(
    country: Optional[str] = Query(None, description="Filter by country (e.g. 'India', 'United States')"),
    release_year: Optional[int] = Query(None, description="Filter by release year (e.g. 2019)"),
    type: Optional[str] = Query(None, description="Filter by title type (e.g. 'Movie', 'TV Show')"),
    rating: Optional[str] = Query(None, description="Filter by rating (e.g. 'TV-MA', 'PG-13')"),
    page: int = Query(1, ge=1, description="Page number, 1-indexed"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Number of items per page"),
    offset: Optional[int] = Query(None, ge=0, description="Optional manual offset override"),
) -> TitleListResponse:
    """Retrieve a paginated list of Netflix titles with optional metadata filters."""
    effective_offset = offset if offset is not None else (page - 1) * page_size
    titles_data, total = query_titles(
        country=country,
        release_year=release_year,
        type_=type,
        rating=rating,
        limit=page_size,
        offset=effective_offset,
    )

    total_pages = math.ceil(total / page_size) if total > 0 else 0

    return TitleListResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        titles=[TitleResponse(**t) for t in titles_data],
    )


@router.get("/titles/{show_id}", response_model=TitleResponse)
def get_title(show_id: int) -> TitleResponse:
    """Retrieve details for a single Netflix title by its show_id."""
    title_data = get_title_by_id(show_id)
    if not title_data:
        raise HTTPException(
            status_code=404,
            detail=f"Title with show_id {show_id} not found in the catalogue.",
        )
    return TitleResponse(**title_data)
