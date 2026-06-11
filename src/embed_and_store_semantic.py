import logging
import sys
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from ingest import _parse_reviews, _course_id_slug
from semantic_chunker import semantic_chunk

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

COLLECTION_NAME = "omshub_reviews_semantic"
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


def _load_semantic_chunks(docs_dir: str, threshold: float) -> list[dict]:
    docs_path = Path(docs_dir)
    all_chunks = []

    for txt_file in sorted(docs_path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        reviews = _parse_reviews(text, txt_file.name)

        course_counter: dict[str, int] = {}
        for review in reviews:
            slug = _course_id_slug(review["course_id"])
            sem = review["semester"]
            yr = review["year"] or "UNKNOWN"
            counter_key = f"{slug}_{sem}_{yr}"
            idx = course_counter.get(counter_key, 0)
            course_counter[counter_key] = idx + 1
            review_id = f"{counter_key}_{idx + 1:03d}"

            chunks = semantic_chunk(
                review["review_text"],
                review_meta=review,
                review_id=review_id,
                threshold=threshold,
            )
            if not chunks:
                chunks = [{
                    "text": review["review_text"],
                    "review_id": review_id,
                    "chunk_index": 0,
                    **{k: review[k] for k in ("course_id", "course_name", "semester",
                                               "year", "date", "workload_hrs",
                                               "difficulty", "overall_rating", "source_file")},
                }]
            all_chunks.extend(chunks)

    return all_chunks


def build(docs_dir: str = "documents", batch_size: int = 64, threshold: float = 0.5) -> int:
    collection = _get_collection()

    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} chunks. Skipping rebuild.")
        return collection.count()

    chunks = _load_semantic_chunks(docs_dir, threshold)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(chunks)} semantic chunks in batches of {batch_size}...")
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

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from embed_and_store import query as query_recursive
    from embed_and_store import build as build_recursive

    # Ensure recursive collection is built too.
    build_recursive()
    total_semantic = build()
    print(f"\nSemantic collection: {total_semantic} chunks")

    test_queries = [
        "How difficult is CS-7641 Machine Learning?",
        "Is CS-6250 good to pair with a hard course?",
        "What are the projects like in CS-6200?",
    ]

    for q in test_queries:
        print("\n" + "=" * 76)
        print(f"QUERY: {q}")
        print("=" * 76)
        print(f"{'RECURSIVE':38} | {'SEMANTIC':36}")
        print("-" * 76)

        r_results = query_recursive(q)
        s_results = query(q)

        for i in range(max(len(r_results), len(s_results))):
            r = r_results[i] if i < len(r_results) else None
            s = s_results[i] if i < len(s_results) else None

            def fmt(res):
                if res is None:
                    return " " * 37
                m = res["metadata"]
                label = f"[{i+1}] {m['course_id']} {m['review_id'][-7:]} d={res['distance']:.3f}"
                return label[:37]

            print(f"{fmt(r):<38}| {fmt(s)}")

        print()
        print("  -- Recursive top result --")
        if r_results:
            print(f"  {r_results[0]['text'][:180].replace(chr(10), ' ')}")
        print("  -- Semantic top result --")
        if s_results:
            print(f"  {s_results[0]['text'][:180].replace(chr(10), ' ')}")
