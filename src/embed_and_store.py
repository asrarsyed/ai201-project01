"""
Milestone 4 — Embedding and Vector Store (Stage 2 of 5)

Loaded after ingest.py. generate.py imports query() from here; the semantic
variant embed_and_store_semantic.py mirrors this interface for its own collection.

Embeds chunk texts from ingest.load_and_chunk() using all-MiniLM-L6-v2
(384-dim, normalize_embeddings=True) and stores vectors + metadata in a local
ChromaDB persistent collection named 'omshub_reviews' (cosine similarity space).
build() is idempotent — skips if the collection is already populated.
query() accepts optional metadata filters (e.g. course_id) and returns the
top-k closest chunks with text, metadata, and cosine distance.
"""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import load_and_chunk

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

COLLECTION_NAME = "omshub_reviews"
CHROMA_DIR = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=CHROMA_DIR)


def _get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _sanitize_metadata(chunk: dict) -> dict:
    """ChromaDB requires str/int/float — no None values."""
    return {
        "review_id": chunk["review_id"],
        "chunk_index": chunk["chunk_index"],
        "course_id": chunk["course_id"],
        "course_name": chunk["course_name"],
        "semester": chunk["semester"],
        "year": chunk["year"] if chunk["year"] is not None else "UNKNOWN",
        "date": chunk["date"],
        "workload_hrs": chunk["workload_hrs"] if chunk["workload_hrs"] is not None else -1.0,
        "difficulty": chunk["difficulty"] if chunk["difficulty"] is not None else 0,
        "overall_rating": chunk["overall_rating"] if chunk["overall_rating"] is not None else 0,
        "source_file": chunk["source_file"],
    }


def build(docs_dir: str = "documents", batch_size: int = 64) -> int:
    """Ingest, embed, and store all chunks. Returns total chunk count."""
    collection = _get_collection()

    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} chunks. Skipping rebuild.")
        return collection.count()

    chunks = load_and_chunk(docs_dir)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        collection.add(
            ids=[f"{c['review_id']}_c{c['chunk_index']}" for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[_sanitize_metadata(c) for c in batch_chunks],
        )

    print(f"Stored {collection.count()} chunks in '{COLLECTION_NAME}'.")
    return collection.count()


def query(text: str, filters: dict | None = None, top_k: int = 5) -> list[dict]:
    """
    Semantic search over stored chunks.

    filters: ChromaDB where-clause dict, e.g.
        {"course_id": "CS-7641"}
        {"$and": [{"semester": "FALL"}, {"difficulty": {"$gte": 4}}]}
    Returns list of dicts with keys: text, metadata, distance.
    """
    collection = _get_collection()

    query_embedding = _model.encode(text, normalize_embeddings=True).tolist()

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "metadata": meta, "distance": dist})

    return output


if __name__ == "__main__":
    total = build()
    print(f"\nTotal chunks in store: {total}")

    test_queries = [
        "How difficult is CS-7641 Machine Learning?",
        "Is CS-6250 good to pair with a hard course?",
        "What are the projects like in CS-6200?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        results = query(q)
        for i, r in enumerate(results, 1):
            m = r["metadata"]
            print(
                f"\n  [{i}] {m['course_id']} ({m['semester']} {m['year']}) "
                f"| review_id={m['review_id']} chunk={m['chunk_index']} "
                f"| dist={r['distance']:.4f}"
            )
            print(
                f"       workload={m['workload_hrs']}h  difficulty={m['difficulty']}/5  overall={m['overall_rating']}/5"
            )
            preview = r["text"][:200].replace("\n", " ")
            print(f"       {preview}...")
