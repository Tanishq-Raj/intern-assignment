"""Retriever module for RAG pipeline.

Computes dense cosine similarity between query and catalogue title embeddings,
with structured metadata filtering (from query intelligence) and keyword fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
from rag.indexer import (
    DEFAULT_INDEX_PATH,
    DEFAULT_MODEL_NAME,
    build_and_save_index,
    get_embedding_model,
    load_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.25

_CACHED_INDEX: dict[str, Any] | None = None

# Fallback demonyms for keyword matching
_DEMONYM_TO_COUNTRY: dict[str, str] = {
    "american": "United States",
    "british": "United Kingdom",
    "dutch": "Netherlands",
    "filipino": "Philippines",
    "south african": "South Africa",
    "new zealand": "New Zealand",
}

# Fallback keywords for type filtering
_TYPE_KEYWORDS: dict[str, str] = {
    "tv show": "TV Show",
    "tv series": "TV Show",
    "series": "TV Show",
    "show": "TV Show",
    "movie": "Movie",
    "film": "Movie",
}


def evaluate_where_filter(item: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a structured where filter against an item's metadata."""
    if not where:
        return True

    for key, condition in where.items():
        if key == "$and":
            if not isinstance(condition, list):
                return False
            if not all(evaluate_where_filter(item, sub) for sub in condition):
                return False

        elif key == "$or":
            if not isinstance(condition, list):
                return False
            if not any(evaluate_where_filter(item, sub) for sub in condition):
                return False

        elif key == "countries":
            item_country = item.get("country")
            if not item_country:
                return False
            target = condition.get("$contains", "") if isinstance(condition, dict) else str(condition)
            if not target:
                continue
            item_countries = [c.strip().lower() for c in str(item_country).split(",")]
            target_lower = target.strip().lower()
            if not (target_lower in item_countries or any(target_lower in c for c in item_countries)):
                return False

        elif key == "genres":
            item_genres = item.get("listed_in")
            if not item_genres:
                return False
            target = condition.get("$contains", "") if isinstance(condition, dict) else str(condition)
            if not target:
                continue
            item_genre_list = [g.strip().lower() for g in str(item_genres).split(",")]
            target_lower = target.strip().lower()
            if not (target_lower in item_genre_list or any(target_lower in g for g in item_genre_list)):
                return False

        elif key == "type":
            item_type = str(item.get("type", "")).strip().lower()
            expected_type = str(condition).strip().lower()
            if item_type != expected_type:
                return False

        elif key == "rating":
            item_rating = str(item.get("rating", "")).strip().lower()
            expected_rating = str(condition).strip().lower()
            if item_rating != expected_rating:
                return False

        elif key == "release_year":
            try:
                item_year = int(item.get("release_year") or 0)
            except (ValueError, TypeError):
                return False

            if isinstance(condition, int):
                if item_year != condition:
                    return False
            elif isinstance(condition, dict):
                for op, val in condition.items():
                    if op == "$eq" and item_year != val:
                        return False
                    elif op == "$gt" and item_year <= val:
                        return False
                    elif op == "$gte" and item_year < val:
                        return False
                    elif op == "$lt" and item_year >= val:
                        return False
                    elif op == "$lte" and item_year > val:
                        return False

    return True


def _match_country(query_lower: str, country_val: str | None) -> bool:
    """Check if a result's country matches any country keyword in the query."""
    if not country_val:
        return False

    item_countries = [c.strip().lower() for c in str(country_val).split(",")]

    for demonym, canonical in _DEMONYM_TO_COUNTRY.items():
        if demonym in query_lower and canonical.lower() in item_countries:
            return True

    for country in item_countries:
        if country in query_lower:
            return True

    return False


def _detect_type_filter(query_lower: str) -> str | None:
    """Detect if the query mentions a content type (TV Show / Movie)."""
    for keyword, type_val in _TYPE_KEYWORDS.items():
        if keyword in query_lower:
            return type_val
    return None


def _apply_keyword_filters(
    results: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """Filter results by matching query keywords against metadata fields (fallback)."""
    q_lower = query.lower()

    has_country_keyword = any(
        c.strip().lower() in q_lower
        for r in results if r.get("country")
        for c in str(r["country"]).split(",")
    ) or any(d in q_lower for d in _DEMONYM_TO_COUNTRY)

    type_filter = _detect_type_filter(q_lower)

    if not has_country_keyword and not type_filter:
        return results

    filtered = []
    for item in results:
        if has_country_keyword and not _match_country(q_lower, item.get("country")):
            continue
        if type_filter and item.get("type", "").lower() != type_filter.lower():
            continue
        filtered.append(item)

    if filtered:
        logger.info("Keyword filter: %d -> %d results", len(results), len(filtered))
        return filtered

    logger.warning("Keyword filter removed all results; returning unfiltered")
    return results


def get_index(index_path: Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    """Get the cached dense index, loading from disk or building if needed."""
    global _CACHED_INDEX
    if _CACHED_INDEX is not None:
        return _CACHED_INDEX

    if not index_path.exists():
        logger.info("Dense index not found at %s. Building new index...", index_path)
        _CACHED_INDEX = build_and_save_index(index_path=index_path)
    else:
        logger.info("Loading RAG dense index from %s...", index_path)
        _CACHED_INDEX = load_index(index_path)

    return _CACHED_INDEX


def reset_index_cache() -> None:
    """Clear memory-cached index (useful during tests)."""
    global _CACHED_INDEX
    _CACHED_INDEX = None


def retrieve_titles(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    index_path: Path = DEFAULT_INDEX_PATH,
    where_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top relevant catalogue titles using dense cosine similarity.

    Applies structured where_filter if provided, or keyword filtering as fallback.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    index_data = get_index(index_path)
    embeddings = index_data["embeddings"]
    metadata = index_data["metadata"]
    model_name = index_data.get("model_name", DEFAULT_MODEL_NAME)

    model = get_embedding_model(model_name)
    query_emb = model.encode([clean_query], normalize_embeddings=True)[0]

    # Compute cosine similarities via dot product of normalized vectors
    similarities = np.dot(embeddings, query_emb)

    if len(similarities) == 0:
        return []

    # Check maximum similarity against threshold
    max_score = float(np.max(similarities))
    if max_score < min_score:
        logger.info(
            "Max similarity %.4f below threshold %.4f for query: '%s'",
            max_score,
            min_score,
            clean_query,
        )
        return []

    # If where_filter is provided, fetch a large candidate pool and filter
    if where_filter:
        fetch_k = min(len(similarities), max(top_k * 10, 100))
        top_indices = np.argsort(similarities)[::-1][:fetch_k]

        filtered_results: list[dict[str, Any]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue
            item = dict(metadata[idx])
            item["score"] = round(score, 4)
            if evaluate_where_filter(item, where_filter):
                filtered_results.append(item)
                if len(filtered_results) >= top_k:
                    break

        if filtered_results:
            logger.info(
                "Structured filter retained %d results for where: %s",
                len(filtered_results),
                where_filter,
            )
            return filtered_results

        logger.warning(
            "Structured filter returned 0 results for where: %s; falling back to unfiltered candidates",
            where_filter,
        )

    # Standard retrieval with keyword filter fallback
    fetch_k = top_k * 4
    top_indices = np.argsort(similarities)[::-1][:fetch_k]

    results: list[dict[str, Any]] = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < min_score:
            continue
        item = dict(metadata[idx])
        item["score"] = round(score, 4)
        results.append(item)

    results = _apply_keyword_filters(results, clean_query)
    return results[:top_k]
