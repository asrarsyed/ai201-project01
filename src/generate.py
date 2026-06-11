"""
Milestone 5 — Grounded Generation (Stage 3 of 5)

Loaded after embed_and_store.py. app.py imports ask() from here.

Retrieves top-k chunks from the vector store, formats them as context with
injected metadata headers (workload, difficulty, overall_rating), and calls the
Groq API (llama-3.3-70b-versatile, temperature=0.2) with a 6-rule system prompt
that enforces answer grounding, explicit refusal on unknown queries, and
disagreement representation. Source attribution is programmatically assembled
from chunk metadata after the LLM call — the model never generates citations.
Auto-detects course IDs in the query via regex and applies a course_id metadata
filter to prevent cross-course semantic bleed.
"""

import os
import re
import sys

from groq import Groq

sys.path.insert(0, os.path.dirname(__file__))
from embed_and_store import query as retrieve

# All course IDs in the corpus, sorted longest-first so e.g. "CS-7643"
# matches before a hypothetical shorter prefix.
_KNOWN_COURSES = sorted(
    [
        "CS-6035",
        "CS-6200",
        "CS-6210",
        "CS-6250",
        "CS-6262",
        "CS-6300",
        "CS-6400",
        "CS-6475",
        "CS-6476",
        "CS-6515",
        "CS-6601",
        "CS-6603",
        "CS-6750",
        "CS-7641",
        "CS-7642",
        "CS-7643",
        "CS-7646",
        "CSE-6040",
        "ISYE-6501",
        "MGT-6203",
    ],
    key=len,
    reverse=True,
)

# Accepts both "CS-7641" and "CS7641" (no hyphen) written in queries.
_COURSE_PATTERNS = [
    (re.compile(re.sub(r"-", r"-?", cid), re.IGNORECASE), cid) for cid in _KNOWN_COURSES
]


def _extract_course_filter(question: str) -> dict | None:
    for pattern, canonical_id in _COURSE_PATTERNS:
        if pattern.search(question):
            return {"course_id": canonical_id}
    return None


_GROQ_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about Georgia Tech OMS courses \
using only student reviews provided to you as context.

Rules you must follow without exception:
1. Answer ONLY from the review chunks provided below. Do not use any outside knowledge.
2. If the provided chunks do not contain enough information to answer the question, \
respond with exactly: "I don't have enough information on that."
3. When reviews disagree or describe different experiences, represent that disagreement \
explicitly. Use phrasing like "some students report... while others note..." — never \
flatten conflicting opinions into a single claim.
4. Do not fabricate citations, course names, or statistics. Every claim must be \
traceable to the provided chunks.
5. Do not add a sources or references section — that is handled separately.
6. Do not reference chunk numbers (e.g. "Chunk 1", "Chunk 2") in your answer. \
Write in natural prose as if summarizing student opinions directly.\
"""


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk["metadata"]
        workload = (
            f"{m['workload_hrs']} hrs/week" if m.get("workload_hrs", -1) != -1 else "not reported"
        )
        difficulty = f"{m['difficulty']}/5" if m.get("difficulty", 0) != 0 else "not reported"
        overall = f"{m['overall_rating']}/5" if m.get("overall_rating", 0) != 0 else "not reported"
        lines.append(
            f"[Chunk {i} | {m['course_id']} {m['course_name']} "
            f"| {m['semester']} {m['year']} | review_id={m['review_id']} "
            f"| workload={workload} difficulty={difficulty} overall={overall}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)


def _format_sources(chunks: list[dict]) -> str:
    seen = set()
    lines = ["Sources:"]
    for chunk in chunks:
        m = chunk["metadata"]
        sem = m["semester"].capitalize() if m["semester"] else "Unknown"
        yr = m["year"]
        rid = m["review_id"]
        entry = f"- {m['course_id']} {m['course_name']} ({sem} {yr}) [{rid}]"
        if entry not in seen:
            seen.add(entry)
            lines.append(entry)
    return "\n".join(lines)


def ask(question: str, filters: dict | None = None, top_k: int = 5) -> dict:
    """
    Retrieve relevant chunks, call Groq, return answer + sources.

    If no filters are provided, auto-detects a course ID in the question
    and applies a course_id filter to prevent cross-course semantic bleed.

    Returns: {"answer": str, "sources": str}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    if filters is None:
        filters = _extract_course_filter(question)

    chunks = retrieve(question, filters=filters, top_k=top_k)
    if not chunks:
        return {
            "answer": "I don't have enough information on that.",
            "sources": "Sources: (none)",
        }

    context = _format_context(chunks)
    user_message = f"Review chunks:\n\n{context}\n\nQuestion: {question}"

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    sources = _format_sources(chunks)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    question = "How many hours per week does CS-7641 Machine Learning typically require?"
    print(f"Question: {question}\n")
    result = ask(question)
    print(result["answer"])
    print()
    print(result["sources"])
