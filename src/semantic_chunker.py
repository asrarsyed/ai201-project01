"""
Stretch Feature — Semantic Chunker (alternative to ingest.py's recursive splitter)

Used by embed_and_store_semantic.py, not by the main pipeline. Not imported by
generate.py or app.py — the main app uses the recursive chunker via embed_and_store.py.

Splits review text at the sentence level by embedding each sentence with
all-MiniLM-L6-v2 and inserting a chunk boundary whenever cosine similarity
between adjacent sentence embeddings drops below a threshold (default 0.5).
Produces topically coherent chunks that naturally separate discussion of projects,
exams, and workload — at the cost of 5.7x more chunks (5,489 vs 962) and slower
ingestion due to per-sentence embedding. Returns the same chunk dict schema as
ingest.load_and_chunk() for drop-in compatibility.
"""

import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentence boundary: punctuation followed by whitespace and a capital letter,
# or end of string. Keeps the delimiter attached to the preceding sentence.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    # Drop empty strings; preserve non-empty fragments (e.g. single sentences
    # with no terminal punctuation).
    return [s.strip() for s in parts if s.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk(
    review_text: str,
    review_meta: dict,
    review_id: str,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Split review_text into semantically coherent chunks.

    Embeds each sentence, inserts a boundary when cosine similarity between
    adjacent sentences drops below `threshold`. Returns list of chunk dicts
    with the same schema as ingest.load_and_chunk().
    """
    sentences = _split_sentences(review_text)

    if not sentences:
        return []

    # Single sentence or very short text — return as-is.
    if len(sentences) == 1:
        return [_make_chunk(review_text, review_id, 0, review_meta)]

    embeddings = _model.encode(sentences, normalize_embeddings=True)

    # Build chunk groups by detecting similarity drops.
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks = []
    for idx, group in enumerate(groups):
        text = " ".join(group)
        if text.strip():
            chunks.append(_make_chunk(text, review_id, idx, review_meta))

    return chunks


def _make_chunk(text: str, review_id: str, chunk_index: int, meta: dict) -> dict:
    return {
        "text": text,
        "review_id": review_id,
        "chunk_index": chunk_index,
        "course_id": meta["course_id"],
        "course_name": meta["course_name"],
        "semester": meta["semester"],
        "year": meta["year"],
        "date": meta["date"],
        "workload_hrs": meta["workload_hrs"],
        "difficulty": meta["difficulty"],
        "overall_rating": meta["overall_rating"],
        "source_file": meta["source_file"],
    }
