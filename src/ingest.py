import logging
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

_model = SentenceTransformer("all-MiniLM-L6-v2")


def token_len(text: str) -> int:
    return len(_model.tokenizer.encode(text, add_special_tokens=False,
                                       truncation=False))


def _parse_workload(raw: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*hrs?/week", raw, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_rating(raw: str) -> int | None:
    m = re.match(r"(\d)/5", raw.strip())
    if m:
        return int(m.group(1))
    return None


def _parse_semester(raw: str):
    parts = raw.strip().split()
    if len(parts) == 2:
        return parts[0].upper(), parts[1]
    return raw.strip().upper(), None


def _course_id_slug(course: str) -> str:
    return course.strip().replace("-", "").upper()


def _parse_reviews(text: str, source_file: str) -> list[dict]:
    blocks = [b.strip() for b in text.split("---") if b.strip()]
    reviews = []
    for block in blocks:
        fields = {}
        lines = block.splitlines()
        current_key = None
        current_val_lines = []

        for line in lines:
            m = re.match(r"^(Course|Course Name|Semester|Date|Workload|Difficulty|Overall|Review):\s*(.*)", line)
            if m:
                if current_key:
                    fields[current_key] = " ".join(current_val_lines).strip()
                current_key = m.group(1)
                current_val_lines = [m.group(2)]
            elif current_key:
                current_val_lines.append(line)

        if current_key:
            fields[current_key] = " ".join(current_val_lines).strip()

        required = {"Course", "Course Name", "Semester", "Date", "Workload", "Difficulty", "Overall", "Review"}
        if not required.issubset(fields):
            continue

        semester_str = fields["Semester"]
        semester, year = _parse_semester(semester_str)

        reviews.append({
            "course_id": fields["Course"].strip(),
            "course_name": fields["Course Name"].strip(),
            "semester": semester,
            "year": year,
            "date": fields["Date"].strip(),
            "workload_hrs": _parse_workload(fields["Workload"]),
            "difficulty": _parse_rating(fields["Difficulty"]),
            "overall_rating": _parse_rating(fields["Overall"]),
            "review_text": fields["Review"].strip(),
            "source_file": source_file,
        })

    return reviews


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if token_len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = None
    remaining_seps = []
    for i, s in enumerate(separators):
        if s in text:
            sep = s
            remaining_seps = separators[i + 1:]
            break

    if sep is None:
        # no separator found: force-split by tokens
        tokens = _model.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = _model.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            start += chunk_size - overlap
        return chunks

    parts = text.split(sep)
    chunks = []
    current_parts = []

    for part in parts:
        candidate = sep.join(current_parts + [part])
        if token_len(candidate) <= chunk_size:
            current_parts.append(part)
        else:
            if current_parts:
                merged = sep.join(current_parts)
                if token_len(merged) > chunk_size and remaining_seps:
                    chunks.extend(_recursive_split(merged, remaining_seps, chunk_size, overlap))
                elif merged.strip():
                    chunks.append(merged)
            current_parts = [part]

    if current_parts:
        merged = sep.join(current_parts)
        if token_len(merged) > chunk_size and remaining_seps:
            chunks.extend(_recursive_split(merged, remaining_seps, chunk_size, overlap))
        elif merged.strip():
            chunks.append(merged)

    # apply overlap: prepend tail of previous chunk to next
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = _model.tokenizer.encode(chunks[i - 1], add_special_tokens=False)
            tail_tokens = prev_tokens[-overlap:]
            tail_text = _model.tokenizer.decode(tail_tokens)
            overlapped.append(tail_text + " " + chunks[i])
        return overlapped

    return chunks


def load_and_chunk(
    docs_dir: str = "documents",
    chunk_size: int = 200,
    overlap: int = 30,
    separators: list[str] | None = None,
) -> list[dict]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    docs_path = Path(docs_dir)
    all_chunks = []

    for txt_file in sorted(docs_path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        reviews = _parse_reviews(text, txt_file.name)

        course_counter: dict[str, int] = {}

        for review in reviews:
            course_id_slug = _course_id_slug(review["course_id"])
            sem = review["semester"]
            yr = review["year"] or "UNKNOWN"
            counter_key = f"{course_id_slug}_{sem}_{yr}"
            idx = course_counter.get(counter_key, 0)
            course_counter[counter_key] = idx + 1
            review_id = f"{counter_key}_{idx + 1:03d}"

            chunks = _recursive_split(review["review_text"], separators, chunk_size, overlap)
            if not chunks:
                chunks = [review["review_text"]]

            for chunk_index, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "review_id": review_id,
                    "chunk_index": chunk_index,
                    "course_id": review["course_id"],
                    "course_name": review["course_name"],
                    "semester": review["semester"],
                    "year": review["year"],
                    "date": review["date"],
                    "workload_hrs": review["workload_hrs"],
                    "difficulty": review["difficulty"],
                    "overall_rating": review["overall_rating"],
                    "source_file": review["source_file"],
                })

    return all_chunks


if __name__ == "__main__":
    import sys

    target_files = {"CS-6250.txt", "CS-7641.txt"}
    docs_path = Path("documents")

    all_chunks = []
    for txt_file in sorted(docs_path.glob("*.txt")):
        if txt_file.name not in target_files:
            continue
        text = txt_file.read_text(encoding="utf-8")
        reviews = _parse_reviews(text, txt_file.name)

        course_counter: dict[str, int] = {}
        for review in reviews:
            course_id_slug = _course_id_slug(review["course_id"])
            sem = review["semester"]
            yr = review["year"] or "UNKNOWN"
            counter_key = f"{course_id_slug}_{sem}_{yr}"
            idx = course_counter.get(counter_key, 0)
            course_counter[counter_key] = idx + 1
            review_id = f"{counter_key}_{idx + 1:03d}"

            chunks = _recursive_split(review["review_text"], ["\n\n", "\n", ". ", " "], 200, 30)
            if not chunks:
                chunks = [review["review_text"]]

            for chunk_index, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "review_id": review_id,
                    "chunk_index": chunk_index,
                    "course_id": review["course_id"],
                    "course_name": review["course_name"],
                    "semester": review["semester"],
                    "year": review["year"],
                    "date": review["date"],
                    "workload_hrs": review["workload_hrs"],
                    "difficulty": review["difficulty"],
                    "overall_rating": review["overall_rating"],
                    "source_file": review["source_file"],
                })

    print(f"Total chunks from CS-6250 + CS-7641: {len(all_chunks)}\n")
    print("=" * 70)
    print("5 SAMPLE CHUNKS")
    print("=" * 70)

    step = max(1, len(all_chunks) // 5)
    samples = [all_chunks[i * step] for i in range(5)]

    for i, chunk in enumerate(samples, 1):
        tlen = token_len(chunk["text"])
        print(f"\n--- Sample {i} ---")
        print(f"review_id    : {chunk['review_id']}")
        print(f"chunk_index  : {chunk['chunk_index']}")
        print(f"course_id    : {chunk['course_id']}")
        print(f"course_name  : {chunk['course_name']}")
        print(f"semester     : {chunk['semester']} {chunk['year']}")
        print(f"workload_hrs : {chunk['workload_hrs']}")
        print(f"difficulty   : {chunk['difficulty']}")
        print(f"overall      : {chunk['overall_rating']}")
        print(f"tokens       : {tlen}")
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"text preview : {preview}...")
