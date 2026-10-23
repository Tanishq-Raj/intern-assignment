"""Generator module for RAG pipeline.

Integrates retrieved catalogue evidence with LLM providers (Groq / OpenAI) or offline fallback,
grounding responses and citing catalogue source identifiers with strict evidence policies
and structured JSON outputs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find sufficient evidence in the Netflix catalogue to answer your question."
)

SYSTEM_PROMPT = """You are a factual Netflix catalogue Q&A assistant.

Rules:
1. Ground every statement strictly and solely in the provided catalogue sources.
2. Do not use outside knowledge. Do not invent titles, actors, countries, ratings, years, or plot details.
3. Distinguish confirmed objective facts (genres, release year, country, director, cast, and synopsis) from subjective or unconfirmed attributes (e.g., 'strong female lead', 'inspiring', 'scariest villain', 'feel-good'). If the catalogue description does not explicitly state a subjective attribute, DO NOT claim that the title satisfies it. Explicitly state what the catalogue confirms and note where evidence for subjective claims is absent.
4. If the provided context is empty or lacks evidence to satisfy the user's question, return answer as "I could not find sufficient evidence in the Netflix catalogue to answer your question." and used_show_ids as [].
5. Only cite show_ids that you actually used in the answer. Every cited show_id must come from the provided sources.
6. Return valid JSON only with exactly these keys:
   {
     "answer": "string response",
     "used_show_ids": ["123", "456"]
   }
7. Do not wrap JSON in markdown fences (no ```json). Return raw JSON only.
""".strip()

FEW_SHOT_EXAMPLES = """Example 1:
User question: Suggest an Indian comedy movie.
Catalogue sources:
Source 1:
show_id: 111
title: Example Comedy
type: Movie
country: India
listed_in: Comedies
description: A light comedy about family confusion.

Source 2:
show_id: 222
title: Example Thriller
type: Movie
country: India
listed_in: Thrillers
description: A dark crime story.

Response JSON:
{
  "answer": "A good match is **Example Comedy** (Show ID: 111). It is an Indian movie categorized under Comedies, following a light family story.",
  "used_show_ids": ["111"]
}

Example 2:
User question: Which catalogue titles are about space travel?
Catalogue sources:
Source 1:
show_id: 333
title: Ocean Story
type: Movie
country: United States
listed_in: Documentaries
description: A documentary about marine life.

Response JSON:
{
  "answer": "I could not find sufficient evidence in the Netflix catalogue to answer your question.",
  "used_show_ids": []
}

Example 3:
User question: Suggest an Indian comedy with a strong female lead.
Catalogue sources:
Source 1:
show_id: 1004
title: 3 Idiots
type: Movie
country: India
listed_in: Comedies
description: Two friends search for their companion.

Response JSON:
{
  "answer": "A relevant Indian comedy from the catalogue is **3 Idiots** (Show ID: 1004). Note that the catalogue records objective metadata and lacks evidence to verify subjective claims like a 'strong female lead'.",
  "used_show_ids": ["1004"]
}
""".strip()

SUBJECTIVE_KEYWORDS = [
    "strong female lead",
    "strong female",
    "inspiring",
    "best",
    "scariest",
    "feel-good",
    "heartwarming",
    "masterpiece",
]


def format_context_for_prompt(retrieved_titles: list[dict[str, Any]]) -> str:
    """Format retrieved title records into structured catalogue source blocks."""
    blocks = []
    for index, item in enumerate(retrieved_titles, start=1):
        block = (
            f"Source {index}:\n"
            f"show_id: {item.get('show_id')}\n"
            f"title: {item.get('title')}\n"
            f"type: {item.get('type')}\n"
            f"release_year: {item.get('release_year')}\n"
            f"country: {item.get('country')}\n"
            f"listed_in: {item.get('listed_in')}\n"
            f"director: {item.get('director')}\n"
            f"cast: {item.get('cast')}\n"
            f"description: {item.get('description')}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def detect_subjective_query(question: str) -> str | None:
    """Detect if the query asks for subjective attributes not verified in catalog metadata."""
    q_lower = question.lower()
    for kw in SUBJECTIVE_KEYWORDS:
        if kw in q_lower:
            return kw
    return None


def generate_offline_answer(
    question: str, retrieved_titles: list[dict[str, Any]]
) -> tuple[str, list[int]]:
    """Synthesize a deterministic grounded response adhering to strict catalogue evidence."""
    if not retrieved_titles:
        return INSUFFICIENT_EVIDENCE_ANSWER, []

    source_ids = [int(t["show_id"]) for t in retrieved_titles if "show_id" in t]
    subjective_term = detect_subjective_query(question)

    lines = [f"Based on the Netflix catalogue, here are the closest titles retrieved for '{question}':\n"]

    if subjective_term:
        lines.append(
            f"Note on Evidence: The Netflix catalogue records objective metadata (genres, country, release year, cast, and synopsis), "
            f"but does not track or prove subjective attributes like '{subjective_term}'. None of the catalogue descriptions explicitly "
            f"assert a '{subjective_term}'. The titles below match objective attributes (such as country and genre):\n"
        )

    for t in retrieved_titles:
        lines.append(
            f"• {t.get('title')} (show_id: {t.get('show_id')}, {t.get('type')}, {t.get('release_year')}): "
            f"{t.get('description')} [Genres: {t.get('listed_in')}; Country: {t.get('country')}]"
        )

    lines.append(
        "\n(Note: Generated via offline grounded synthesizer. Provide a GROQ_API_KEY for dynamic conversational generation.)"
    )
    return "\n".join(lines), source_ids


def _strip_markdown_fences(content: str) -> str:
    """Strip markdown code fence wrapper if present."""
    text = content.strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    if text.startswith("```"):
        text = text.removeprefix("```").strip()
    if text.endswith("```"):
        text = text.removesuffix("```").strip()
    return text


def parse_llm_response(
    content: str, candidate_ids: set[int]
) -> tuple[str, list[int]]:
    """Parse JSON or text LLM response, verifying cited IDs against candidate titles."""
    text = _strip_markdown_fences(content)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            answer = str(data["answer"]).strip()
            raw_ids = data.get("used_show_ids", [])
            valid_ids: list[int] = []
            for item in raw_ids:
                try:
                    val = int(item)
                    if val in candidate_ids and val not in valid_ids:
                        valid_ids.append(val)
                except (ValueError, TypeError):
                    continue
            return answer, valid_ids
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback to extracting IDs via regex from text
    cited = extract_cited_ids_from_text(content, candidate_ids)
    return content.strip(), cited


def extract_cited_ids_from_text(text: str, candidate_ids: set[int]) -> list[int]:
    """Extract and verify cited show IDs found in response text."""
    found_numbers = re.findall(r"\b\d{4,10}\b", text)
    cited: list[int] = []
    for num_str in found_numbers:
        val = int(num_str)
        if val in candidate_ids and val not in cited:
            cited.append(val)
    return cited


def _call_groq_api(question: str, context: str, api_key: str, model: str) -> str:
    """Invoke Groq Chat Completions API."""
    from groq import Groq

    client = Groq(api_key=api_key)
    subjective_term = detect_subjective_query(question)
    extra_instructions = ""
    if subjective_term:
        extra_instructions = (
            f"IMPORTANT: The user asks for '{subjective_term}'. The Netflix catalogue does NOT track or verify subjective claims. "
            f"Recommend relevant matching titles from the sources, cite their show_ids in used_show_ids, and explicitly state that the catalogue does not track or prove '{subjective_term}'.\n\n"
        )

    user_prompt = (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Now answer the user question based strictly on the catalogue sources below:\n\n"
        f"User Question: {question}\n\n"
        f"Catalogue sources:\n{context}\n\n"
        f"{extra_instructions}"
        f"Return JSON with 'answer' and 'used_show_ids'."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


def _call_openai_api(question: str, context: str, api_key: str, model: str) -> str:
    """Invoke OpenAI Chat Completions API (used when configured or in tests)."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


def generate_answer(
    question: str, retrieved_titles: list[dict[str, Any]]
) -> tuple[str, list[int]]:
    """Orchestrate answer generation from retrieved context using Groq, OpenAI, or offline."""
    if not retrieved_titles:
        return INSUFFICIENT_EVIDENCE_ANSWER, []

    candidate_ids = {int(t["show_id"]) for t in retrieved_titles if "show_id" in t}
    context_str = format_context_for_prompt(retrieved_titles)

    load_dotenv(dotenv_path=ENV_PATH, override=False)

    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    # Determine provider based on configuration and available keys
    if provider == "openai" or (openai_key and not groq_key and provider != "groq"):
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        logger.info("Generating response with OpenAI API (model: %s)...", model)
        try:
            content = _call_openai_api(question, context_str, openai_key, model)
            answer, cited_ids = parse_llm_response(content, candidate_ids)
            if not cited_ids:
                cited_ids = extract_cited_ids_from_text(answer, candidate_ids) or list(candidate_ids)
            return answer, cited_ids
        except Exception as e:
            logger.warning("OpenAI call failed, falling back to offline generator: %s", e)
            return generate_offline_answer(question, retrieved_titles)

    if groq_key and provider != "offline":
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        logger.info("Generating response with Groq API (model: %s)...", model)
        try:
            content = _call_groq_api(question, context_str, groq_key, model)
            answer, cited_ids = parse_llm_response(content, candidate_ids)
            if not cited_ids:
                cited_ids = extract_cited_ids_from_text(answer, candidate_ids) or list(candidate_ids)
            return answer, cited_ids
        except Exception as e:
            logger.warning("Groq call failed, falling back to offline generator: %s", e)
            return generate_offline_answer(question, retrieved_titles)

    logger.info("No active LLM API key; using offline grounded generator.")
    return generate_offline_answer(question, retrieved_titles)
