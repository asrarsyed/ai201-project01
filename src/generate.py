import os
import sys

from groq import Groq

sys.path.insert(0, os.path.dirname(__file__))
from embed_and_store import query as retrieve

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
5. Do not add a sources or references section — that is handled separately.\
"""


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk["metadata"]
        lines.append(
            f"[Chunk {i} | {m['course_id']} {m['course_name']} "
            f"| {m['semester']} {m['year']} | review_id={m['review_id']}]\n"
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

    Returns: {"answer": str, "sources": str}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

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
