"""RAG Q&A route handlers for catalogue grounded questions."""

from __future__ import annotations

import logging
from fastapi import APIRouter
from api.models import AskRequest, AskResponse, SourceItem
from rag.generator import generate_answer
from rag.query_intelligence import create_search_plan
from rag.retriever import retrieve_titles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG Q&A"])


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """Answer questions about the Netflix catalogue using query-intelligent grounded RAG retrieval."""
    logger.info("Received question: '%s'", payload.question)

    # 1. Query Intelligence: parse question into semantic query & structured metadata filters
    search_plan = create_search_plan(payload.question)

    # 2. Vector Search + Post-retrieval Metadata Filtering
    retrieved_items = retrieve_titles(
        query=search_plan.semantic_query,
        top_k=5,
        where_filter=search_plan.where,
    )

    # Fallback to direct question search if structured filter yielded no candidates
    if not retrieved_items and search_plan.where is not None:
        logger.info("Structured filter produced 0 candidates; falling back to direct question search")
        retrieved_items = retrieve_titles(query=payload.question, top_k=5)

    # 3. Grounded Generation
    answer, cited_ids = generate_answer(payload.question, retrieved_items)

    # 4. Filter sources strictly to the cited/used titles
    item_by_id = {item["show_id"]: item for item in retrieved_items}
    valid_cited_ids = [sid for sid in cited_ids if sid in item_by_id]

    sources = [
        SourceItem(
            show_id=item_by_id[sid]["show_id"],
            title=item_by_id[sid]["title"],
            type=item_by_id[sid]["type"],
            release_year=item_by_id[sid]["release_year"],
            country=item_by_id[sid].get("country"),
            rating=item_by_id[sid].get("rating"),
            score=item_by_id[sid].get("score"),
        )
        for sid in valid_cited_ids
    ]

    return AskResponse(
        question=payload.question,
        answer=answer,
        show_ids=valid_cited_ids,
        sources=sources,
    )
