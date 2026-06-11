# The Unofficial Guide — Project 1

---

## Domain

Student course reviews for Georgia Tech's Online Master of Science (OMS) programs, sourced from [OMSHub](https://www.omshub.org/), a student-run review platform. Official course descriptions list topics and prerequisites but reveal nothing about actual workload, exam format, which assignments are hardest, or whether a course is a good fit to pair alongside a heavier one. Students typically piece this together from Reddit threads, Discord servers, and word of mouth. This system makes that collective knowledge searchable through natural language queries grounded entirely in real student reviews across 20 courses.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | CS-6035 Introduction to Information Security | OMSHub Course Reviews | [omshub.org/course/CS-6035](https://www.omshub.org/course/CS-6035) |
| 2 | CS-6200 Introduction to Operating Systems | OMSHub Course Reviews | [omshub.org/course/CS-6200](https://www.omshub.org/course/CS-6200) |
| 3 | CS-6210 Advanced Operating Systems | OMSHub Course Reviews | [omshub.org/course/CS-6210](https://www.omshub.org/course/CS-6210) |
| 4 | CS-6250 Computer Networks | OMSHub Course Reviews | [omshub.org/course/CS-6250](https://www.omshub.org/course/CS-6250) |
| 5 | CS-6262 Network Security | OMSHub Course Reviews | [omshub.org/course/CS-6262](https://www.omshub.org/course/CS-6262) |
| 6 | CS-6300 Software Development Process | OMSHub Course Reviews | [omshub.org/course/CS-6300](https://www.omshub.org/course/CS-6300) |
| 7 | CS-6400 Database Systems Concepts and Design | OMSHub Course Reviews | [omshub.org/course/CS-6400](https://www.omshub.org/course/CS-6400) |
| 8 | CS-6475 Computational Photography | OMSHub Course Reviews | [omshub.org/course/CS-6475](https://www.omshub.org/course/CS-6475) |
| 9 | CS-6476 Computer Vision | OMSHub Course Reviews | [omshub.org/course/CS-6476](https://www.omshub.org/course/CS-6476) |
| 10 | CS-6515 Intro to Graduate Algorithms | OMSHub Course Reviews | [omshub.org/course/CS-6515](https://www.omshub.org/course/CS-6515) |
| 11 | CS-6601 Artificial Intelligence | OMSHub Course Reviews | [omshub.org/course/CS-6601](https://www.omshub.org/course/CS-6601) |
| 12 | CS-6603 AI, Ethics, and Society | OMSHub Course Reviews | [omshub.org/course/CS-6603](https://www.omshub.org/course/CS-6603) |
| 13 | CS-6750 Human-Computer Interaction | OMSHub Course Reviews | [omshub.org/course/CS-6750](https://www.omshub.org/course/CS-6750) |
| 14 | CS-7641 Machine Learning | OMSHub Course Reviews | [omshub.org/course/CS-7641](https://www.omshub.org/course/CS-7641) |
| 15 | CS-7642 Reinforcement Learning | OMSHub Course Reviews | [omshub.org/course/CS-7642](https://www.omshub.org/course/CS-7642) |
| 16 | CS-7643 Deep Learning | OMSHub Course Reviews | [omshub.org/course/CS-7643](https://www.omshub.org/course/CS-7643) |
| 17 | CS-7646 Machine Learning for Trading | OMSHub Course Reviews | [omshub.org/course/CS-7646](https://www.omshub.org/course/CS-7646) |
| 18 | CSE-6040 Computing for Data Analysis | OMSHub Course Reviews | [omshub.org/course/CSE-6040](https://www.omshub.org/course/CSE-6040) |
| 19 | ISYE-6501 Intro to Analytics Modeling | OMSHub Course Reviews | [omshub.org/course/ISYE-6501](https://www.omshub.org/course/ISYE-6501) |
| 20 | MGT-6203 Data Analytics in Business | OMSHub Course Reviews | [omshub.org/course/MGT-6203](https://www.omshub.org/course/MGT-6203) |

---

## Chunking Strategy

**Chunk size:** 200 tokens (measured by the `all-MiniLM-L6-v2` tokenizer itself)

**Overlap:** 30 tokens prepended from the tail of the previous chunk

**Why these choices fit your documents:**

OMSHub reviews range from under 100 words to over 2,500 words — a variance of roughly 25x. The primary hard constraint is the embedding model: `all-MiniLM-L6-v2` silently truncates any input above 256 tokens, so content past that limit simply disappears during embedding. A 200-token target leaves a safe buffer below that ceiling. Short reviews (under 200 tokens) are stored as a single chunk — splitting a 150-word review would produce fragments too small to carry meaning. Long reviews get split into 8–12 focused chunks, which is intentional: a 2,500-word review typically covers projects, exams, workload, and professor quality in separate paragraphs. Smaller focused chunks mean a query about "exam difficulty" retrieves the exam paragraph specifically, rather than a diluted whole-review embedding.

The 30-token overlap prevents meaning loss at chunk boundaries. Reviews often use referential language mid-thought ("but that said, the final project..." loses its antecedent without the preceding sentence). 30 tokens carries that context without duplicating significant content across chunks.

A custom recursive Python splitter with separators `["\n\n", "\n", ". ", " "]` attempts to split on paragraph breaks first, then newlines, then sentence boundaries, then words, avoiding mid-sentence cuts wherever possible. Token counting uses `all-MiniLM-L6-v2`'s own tokenizer via a `token_len()` helper, so "200 tokens" means exactly 200 tokens as the model counts them.

Every chunk inherits its parent review's metadata: `course_id`, `course_name`, `semester`, `year`, `difficulty`, `workload_hrs`, `overall_rating`, `review_id` (format: `CS6250_SPRING_2022_001`), and `chunk_index`. Metadata is also injected into each chunk's header line when passed to the LLM, so workload and difficulty numbers — which appear in the structured review header, not the review body — are always visible in context.

**Final chunk count:** 962 chunks across all 20 documents. No chunk exceeds 231 tokens (well within the 256-token model limit).

---

## Sample Chunks

Below are 5 labeled sample chunks from the two-file test run (`CS-6250.txt` and `CS-7641.txt`).

**Sample 1** — `CS6250_FALL_2025_001`, chunk 0 (173 tokens) — `CS-6250.txt`
```
CS-6250 (Computer Networks) has some real strengths, especially the support
system around it. The TAs are genuinely helpful and responsive—big shout-out
to Erik, who is the reason I am commenting here—and that support makes a huge
difference when you're working through the tougher parts of the course.
```
*Metadata: workload=12.0 hrs/week, difficulty=3/5, overall=3/5*

**Sample 2** — `CS6250_SUMMER_2023_003`, chunk 11 (221 tokens) — `CS-6250.txt`
```
little - to - no required time allocation for that particular week (i.e., if
completed project and quiz ahead of schedule by that point). Course
Deliverables Quizzes: In general, quizzes were released weekly, so it was not
possible to work ahead on those (i.e., as one quiz was due, the next week's...)
```
*Metadata: workload=10.9 hrs/week, difficulty=2/5, overall=2/5*

**Sample 3** — `CS6250_SPRING_2022_004`, chunk 0 (184 tokens) — `CS-6250.txt`
```
I have a background in computer science and I took this course to refresh some
of my CN concepts. This is an easy course and a great one to pair with a
medium/hard course. The projects are fun and you get enough time for each of
them.
```
*Metadata: workload=6.0 hrs/week, difficulty=2/5, overall=4/5*

**Sample 4** — `CS6250_FALL_2021_003`, chunk 2 (77 tokens) — `CS-6250.txt`
```
many interesting discussions happening on ed. honestly, i don't think that
this was worth the money. i wish some of the topics were covered better. I
have mistakenly assumed that since this is a graduate level course I'll
definitely learn something from it, but to me it seemed more like an intro...
```
*Metadata: workload=5.0 hrs/week, difficulty=1/5, overall=3/5*

**Sample 5** — `CS7641_FALL_2022_002`, chunk 3 (224 tokens) — `CS-7641.txt`
```
set with categorical attributes or some unstructured data like text or image
- my advice would be to only choose tabular data with all continuous attributes.
For MDP, you should likely choose Frozen-Lake - other choices are BlackJack,
Forest Management, RiverSwim, etc. Kaggle is great for running a...
```
*Metadata: workload=30.0 hrs/week, difficulty=5/5, overall=5/5*

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (local, no API key). Produces 384-dimensional embeddings. Vector store is ChromaDB with cosine similarity (`hnsw:space=cosine`), `normalize_embeddings=True`.

**Production tradeoff reflection:**

`all-MiniLM-L6-v2` was trained on general web text, not academic or student-written prose. Informal, fragmented review language embeds differently from clean natural-language queries, which is why cosine distances in this corpus cluster in the 0.43–0.52 range even for clearly relevant results — a known gap. In production, several tradeoffs would be reconsidered:

- **Context length:** The 256-token hard truncation is a real constraint. OpenAI's `text-embedding-3-small` or Cohere's `embed-english-v3.0` support much longer inputs, reducing or eliminating the need to split short reviews at all.
- **Domain specificity:** A model fine-tuned on academic or student-written text would produce better similarity scores for domain vocabulary ("proctored exam," "project rubric," "office hours," grade-component jargon).
- **Latency vs. accuracy:** Local embedding is fast and free but less accurate. For a production system, the accuracy improvement from a larger API-based model may justify the cost — especially for a corpus where informal writing style is the main source of retrieval noise.
- **Multilingual support:** Not a concern for this corpus (English-only), but models like `paraphrase-multilingual-MiniLM-L12-v2` would matter for an international student platform.

---

## Retrieval Test Results

All queries use `top_k=5`. Queries naming a specific course have a `course_id` metadata filter automatically applied by `generate.py`'s regex extractor. Distance is cosine distance (lower = more similar).

### Query 1: "How many hours per week does CS-7641 Machine Learning typically require?"

*(Auto-filter: `course_id=CS-7641` applied)*

| Rank | Course | Review ID | Distance |
|------|--------|-----------|----------|
| 1 | CS-7641 Machine Learning | CS7641_SPRING_2024_001 | 0.488 |
| 2 | CS-7641 Machine Learning | CS7641_SPRING_2022_004 | 0.502 |
| 3 | CS-7641 Machine Learning | CS7641_FALL_2022_002 | 0.511 |
| 4 | CS-7641 Machine Learning | CS7641_FALL_2021_003 | 0.521 |
| 5 | CS-7641 Machine Learning | CS7641_FALL_2022_001 | 0.534 |

**Why these chunks are relevant:** All 5 are CS-7641 reviews. The workload numbers (8, 10, 22, 25, 30 hrs/week) come from `workload_hrs` metadata injected into each chunk header — they do not appear in the review body text. Without metadata injection, the LLM would have said "I don't have enough information." The high distances (0.49–0.53) reflect all-MiniLM's general-web training not aligning well with informal review prose, but the correct chunks are still retrieved because the course filter eliminates cross-course competition.

### Query 2: "Is CS-6250 Computer Networks a good course to pair with a more difficult course?"

*(Auto-filter: `course_id=CS-6250` applied)*

| Rank | Course | Review ID | Distance |
|------|--------|-----------|----------|
| 1 | CS-6250 Computer Networks | CS6250_FALL_2025_001 | 0.251 |
| 2 | CS-6250 Computer Networks | CS6250_FALL_2022_001 | 0.302 |
| 3 | CS-6250 Computer Networks | CS6250_SPRING_2022_002 | 0.318 |
| 4 | CS-6250 Computer Networks | CS6250_FALL_2021_003 | 0.344 |
| 5 | CS-6250 Computer Networks | CS6250_SPRING_2022_005 | 0.353 |

**Why these chunks are relevant:** Distances here (0.25–0.35) are significantly lower than Query 1 because CS-6250 reviews frequently use pairing and difficulty language ("easy course," "pair with," "manageable workload") that semantically overlaps well with the query. Rank 3 contains the exact sentence: "This is an easy course and a great one to pair with a medium/hard course." Rank 5 says: "If you want an easy course, this is the one to take." The model retrieves multiple perspectives on pairing, including one student who paired it with Game AI successfully.

### Query 3: "What do students say about the projects in CS-6200 Introduction to Operating Systems?"

*(Auto-filter: `course_id=CS-6200` applied)*

| Rank | Course | Review ID | Distance |
|------|--------|-----------|----------|
| 1 | CS-6200 Introduction to Operating Systems | CS6200_FALL_2022_003 | 0.366 |
| 2 | CS-6200 Introduction to Operating Systems | CS6200_SPRING_2023_001 | 0.370 |
| 3 | CS-6200 Introduction to Operating Systems | CS6200_FALL_2022_003 | 0.393 |
| 4 | CS-6200 Introduction to Operating Systems | CS6200_SPRING_2023_001 | 0.403 |
| 5 | CS-6262 Network Security | CS6262_SPRING_2022_001 | 0.406 |

Only 3 unique CS-6200 source reviews returned (rank 5 is an off-course result). The answer quality is still good because the 3 CS-6200 chunks cover the main project claims: C/C++ system programming, client-server architecture, multithreading/IPC/RPC assignments, and difficulty of parsing project instructions.

---

## Grounded Generation

**System prompt grounding instruction:**

The system prompt includes six explicit rules. The core grounding constraints are:

```
1. Answer ONLY from the review chunks provided below. Do not use any outside knowledge.
2. If the provided chunks do not contain enough information to answer the question,
   respond with exactly: "I don't have enough information on that."
3. When reviews disagree or describe different experiences, represent that disagreement
   explicitly. Use phrasing like "some students report... while others note..." —
   never flatten conflicting opinions into a single claim.
4. Do not fabricate citations, course names, or statistics. Every claim must be
   traceable to the provided chunks.
5. Do not add a sources or references section — that is handled separately.
6. Do not reference chunk numbers (e.g. "Chunk 1", "Chunk 2") in your answer.
   Write in natural prose as if summarizing student opinions directly.
```

Rules 1 and 2 enforce factual grounding. Rule 3 handles the contradiction problem specific to student review corpora — popular courses have bimodal reviews. Rule 6 was added after discovering the LLM leaked internal chunk indices ("as stated in Chunk 2...") into answer prose.

**Structural grounding mechanisms:**

1. Each chunk sent to the LLM includes a header with parsed metadata: `[Chunk N | CS-7641 Machine Learning | FALL 2022 | review_id=CS7641_FALL_2022_002 | workload=30.0 hrs/week difficulty=5/5 overall=5/5]`. This injects workload and difficulty numbers that exist only in the structured review header, not the review body text — without this, the LLM cannot answer workload questions accurately.

2. Auto course-ID filter: `generate.py` applies a regex against `_KNOWN_COURSES` to detect course mentions in the query and applies a `course_id` metadata filter before retrieval. This eliminates cross-course semantic bleed (e.g. a CS-6475 chunk ranking above CS-7641 chunks for a CS-7641 workload query because both mention "hrs/week" densely).

**How source attribution is surfaced in the response:**

Sources are programmatically assembled from chunk metadata by `_format_sources()` and appended to the response string after the LLM call — the LLM never writes citations itself. Format: `- CS-7641 Machine Learning (Fall 2022) [CS7641_FALL_2022_002]`. Deduplication ensures each unique review appears once even if multiple chunks from the same review were retrieved.

---

## Example Responses

### Example 1 — CS-7641 workload (Q1)

**Query:** "How many hours per week does CS-7641 Machine Learning typically require?"

**Response:**
> The number of hours per week required for CS-7641 Machine Learning varies significantly according to student reviews. Some students report a workload of 8.0 hours/week, while others note workloads of 10.0 hours/week, 22.0 hours/week, 25.0 hours/week, and even as high as 30.0 hours/week. This wide range suggests that the actual time commitment can differ substantially from one student to another.

**Sources:**
```
Sources:
- CS-7641 Machine Learning (Spring 2024) [CS7641_SPRING_2024_001]
- CS-7641 Machine Learning (Spring 2022) [CS7641_SPRING_2022_004]
- CS-7641 Machine Learning (Fall 2022) [CS7641_FALL_2022_002]
- CS-7641 Machine Learning (Fall 2021) [CS7641_FALL_2021_003]
- CS-7641 Machine Learning (Fall 2022) [CS7641_FALL_2022_001]
```

### Example 2 — CS-6250 pairing (Q2)

**Query:** "Is CS-6250 Computer Networks a good course to pair with a more difficult course?"

**Response:**
> Some students report that CS-6250 Computer Networks is a good course to pair with another class, as it is considered relatively easy and has a manageable workload. For example, one student paired it with Game AI and found it to be a solid combination. However, others note that if you're already familiar with networking, you might not get too much out of this course, so it may depend on your individual background and goals. Additionally, the workload and difficulty of the course can vary, with some students reporting it requires around 4–7 hours of work per week and a difficulty of 1–2 out of 5, while others report it requires around 12 hours of work per week and a difficulty of 3 out of 5.

**Sources:**
```
Sources:
- CS-6250 Computer Networks (Fall 2025) [CS6250_FALL_2025_001]
- CS-6250 Computer Networks (Fall 2022) [CS6250_FALL_2022_001]
- CS-6250 Computer Networks (Spring 2022) [CS6250_SPRING_2022_002]
- CS-6250 Computer Networks (Fall 2021) [CS6250_FALL_2021_003]
- CS-6250 Computer Networks (Spring 2022) [CS6250_SPRING_2022_005]
```

### Example 3 — Out-of-scope query (refusal)

**Query:** "What is the best pizza place near Georgia Tech campus?"

**Response:**
> I don't have enough information on that.

The system returns the refusal string exactly as specified in rule 2. Retrieval still runs (it returned unrelated course review chunks about business analytics and photography), but the LLM correctly identifies that none of the context is relevant to the question.

---

## Query Interface

**Input field:** "Ask about any OMS course" — a 2-line freeform text box with placeholder text "e.g. How hard is CS-7641 Machine Learning?" Submit via the "Ask" button or Enter key.

**Output fields:**
- "Answer" — 10-line read-only text box displaying the grounded LLM response.
- "Retrieved from" — 4-line read-only text box displaying the programmatically assembled sources block (course name, semester, review ID).

**Sample interaction transcript:**

```
User input:
  Is CS-6603 AI, Ethics, and Society considered an easy course?

Answer:
  Some students report that CS-6603 AI, Ethics, and Society is an easy course,
  with one student noting it is the "second easiest class" they've taken as part
  of OMSCS, and another stating that the assignments are "very straightforward"
  and the tests are "fairly easy" if you watch the lectures. Additionally, several
  students mention that the course is not challenging, with workload estimates
  ranging from 4 to 11.5 hours per week and difficulty ratings generally being on
  the lower end. However, others note that while the course may be easy, it has
  some poorly designed assignments and lacks clarity in some areas, which can be
  frustrating. Overall, the consensus among students is that the course is not
  particularly difficult, but opinions on its value and quality vary.

Retrieved from:
  Sources:
  - CS-6603 AI, Ethics, and Society (Spring 2022) [CS6603_SPRING_2022_003]
  - CS-6603 AI, Ethics, and Society (Spring 2024) [CS6603_SPRING_2024_001]
  - CS-6603 AI, Ethics, and Society (Fall 2022) [CS6603_FALL_2022_001]
  - CS-6603 AI, Ethics, and Society (Summer 2024) [CS6603_SUMMER_2024_001]
  - CS-6603 AI, Ethics, and Society (Fall 2022) [CS6603_FALL_2022_003]
```

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How many hours per week does CS-7641 Machine Learning typically require? | 15–20+ hrs/week consistently reported | 8–30 hrs/week range across reviewers; surfaced variation correctly | Relevant (course filter applied) | Accurate |
| 2 | Is CS-6250 Computer Networks a good course to pair with a more difficult course? | Yes, easy/manageable, ~6–7 hrs/week | Yes, 4–7 hrs/week at low difficulty; acknowledged 12 hrs/week outlier; mentioned Game AI pairing | Relevant | Accurate |
| 3 | What do students say about the projects in CS-6200 Introduction to Operating Systems? | Challenging C/C++ systems programming, bulk of course difficulty | C/C++ client-server, multithreading/IPC/RPC; noted instruction-parsing difficulty; only 3 unique sources returned | Partially relevant (rank 5 was off-course) | Accurate |
| 4 | How difficult are the exams in CS-6515 Intro to Graduate Algorithms? | Proctored, algorithm-heavy, among hardest in OMS | Surfaced difficulty and subjectivity of grading; cited study strategies and 30 hrs/week prep; missed proctored exam framing | Relevant | Partially accurate |
| 5 | Is CS-6603 AI, Ethics, and Society considered an easy course? | Generally yes — light workload, discussion/writing focused | Yes, seconded by multiple reviewers; 4–11.5 hrs/week; noted poor assignment design as a nuance | Relevant | Accurate |

---

## Failure Case Analysis

**Question that failed:** "How difficult are the exams in CS-6515 Intro to Graduate Algorithms?" (Q4)

**What the system returned:** The response correctly described exams as challenging and noted grading subjectivity, preparation strategies, and the value of studying dynamic programming and graph algorithms. It did not mention that CS-6515 exams are proctored — the most defining signal in the expected answer and the primary reason the course is considered among the hardest in the program.

**Root cause (tied to a specific pipeline stage):** The failure is in the retrieval stage, not generation. The query "How difficult are the exams" retrieves chunks semantically similar to difficulty and exam language in general — chunks about study strategies, grading rubrics, and preparation tips rank highly because they use the same vocabulary. The chunks that specifically mention the proctored format and the formal, algorithm-proof style of the exams did not rank in the top 5. The LLM answered faithfully from what it received; it simply never received the relevant context.

**What you would change to fix it:** Increase `top_k` from 5 to 8 or 10 for queries about exams — the proctored format is mentioned in fewer reviews and ranks outside the top 5. Alternatively, a targeted metadata filter or hybrid keyword search that boosts chunks containing "proctored" or "timed" would surface this signal more reliably.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The spec's chunk size rationale — specifically the argument that 200 tokens is the right target because `all-MiniLM-L6-v2` silently truncates at 256 — shaped the implementation from the start. Without that constraint written into `planning.md`, the natural default would have been a character-based splitter with a rough size estimate. Instead, the implementation used the model's own tokenizer as the length function (`token_len()` via `_model.tokenizer.encode()`), which guaranteed "200 tokens" meant exactly what the embedding model would count. That precision made the max-token-per-chunk verification (`max: 231 tokens`) meaningful.

**One way your implementation diverged from the spec, and why:**

The spec described source attribution as `- COURSE_NAME (Semester Year) [review_id]` with the LLM generating this block. During implementation, it became clear that leaving citation generation to the LLM creates a hallucination risk — the model might invent or misattribute review IDs. The final implementation programmatically assembles the sources block from chunk metadata after the LLM call, with the LLM explicitly instructed not to generate a sources section at all (rule 5 in the system prompt). The spec's intent was correct; the mechanism changed to close the hallucination vector.

---

## AI Usage

**Instance 1 — Ingestion and chunking pipeline**

- *What I gave the AI:* The full Chunking Strategy section from `planning.md` (chunk size rationale, overlap reasoning, separator list, `review_id` format spec, metadata schema), plus a sample from the `.txt` file format showing the `Course:` / `Semester:` / `Review:` field structure. I also specified that token counting must use `all-MiniLM-L6-v2`'s own tokenizer, not a character approximation.
- *What it produced:* `src/ingest.py` with `_parse_reviews()`, `_recursive_split()`, `load_and_chunk()`, and the `token_len()` helper using `model.tokenizer.encode()`. The structure matched the spec closely.
- *What I changed or overrode:* The initial implementation used `warnings.catch_warnings()` to suppress a tokenizer length warning from transformers. This failed because the warning came from `logger.warning()` (a Python logging call), not `warnings.warn()` — `catch_warnings()` cannot intercept logging. I directed the AI to fix this by setting `logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)` instead, which correctly suppressed the warning.

**Instance 2 — Grounded generation and course-ID filter**

- *What I gave the AI:* The system prompt requirements from `planning.md` (ground-only, explicit refusal on unknown, conflict acknowledgment), the Groq API model name (`llama-3.3-70b-versatile`), and the requirement that citations be programmatically assembled — not LLM-generated. I also directed it not to use `python-dotenv` since the environment uses `direnv` to load `GROQ_API_KEY` directly.
- *What it produced:* `src/generate.py` with a 5-rule system prompt, `_format_context()` injecting metadata headers, `_format_sources()` deduplicating by entry string, and `ask()` as the main interface.
- *What I changed or overrode:* The initial version did not auto-detect course IDs in queries, causing cross-course semantic bleed (CS-6475 ranked above CS-7641 chunks for a CS-7641 workload query). I directed the AI to add `_extract_course_filter()` — a regex over `_KNOWN_COURSES` sorted longest-first — to automatically apply a `course_id` filter when a known course is mentioned in the query. I also directed it to add rule 6 to the system prompt after observing the LLM writing "as stated in Chunk 2" in answer prose.
