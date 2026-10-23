"""Pydantic schemas for the Netflix Catalog API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class TitleResponse(BaseModel):
    """Schema representing a single Netflix title."""

    show_id: int
    type: str
    title: str
    director: str | None = None
    cast: str | None = None
    country: str | None = None
    date_added: str | None = None
    date_added_iso: str | None = None
    release_year: int
    rating: str | None = None
    duration: str
    listed_in: str
    description: str


class TitleListResponse(BaseModel):
    """Schema representing a paginated list of titles."""

    total: int
    page: int
    page_size: int
    total_pages: int
    titles: list[TitleResponse]


class CountryStat(BaseModel):
    """Schema for country frequency counts."""

    country: str
    count: int


class StatsResponse(BaseModel):
    """Schema for catalogue summary statistics."""

    total_titles: int
    count_by_type: dict[str, int]
    count_by_country: list[CountryStat]


class AskRequest(BaseModel):
    """Schema for the RAG question request."""

    question: str = Field(..., min_length=2, description="Natural language question about the catalogue")


class SourceItem(BaseModel):
    """Schema for a catalogue evidence title cited by RAG."""

    show_id: int
    title: str
    type: str
    release_year: int
    country: str | None = None
    rating: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    """Schema for grounded RAG answer with source citations."""

    question: str
    answer: str
    show_ids: list[int]
    sources: list[SourceItem]
