# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation -- the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

I chose student course reviews for Georgia Tech's Online Master of Science programs pulled from OMSHub, a student-run review platform. Official course descriptions explain what a course covers, but they don't reveal what the workload is actually like, how many hours students typically spend each week, or which aspects of a course are considered most challenging. Students often have to piece together this information from review sites, Reddit discussions, and word of mouth.

To make this knowledge easier to access, I collected approximately 400 of the most recent reviews across 20 courses as the foundation for my system that allows users to ask questions and receive answers grounded in real student experiences. While this project focuses on 20 courses, the pipeline can easily be expanded in the future to include all OMSHub courses with available reviews.

> [!NOTE]
> TLDR: Student reviews of courses for OMS programs is my domain; [OMSHUB](https://www.omshub.org/) is the source for my documents!

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Type | Local file | Original URL |
|---|--------|------|:----------:|:------------:|
| 1 | CS-6035 Introduction to Information Security | OMSHub Course Reviews | [File](documents/CS-6035.txt) | [URL](https://www.omshub.org/course/CS-6035) |
| 2 | CS-7646 Machine Learning for Trading | OMSHub Course Reviews | [File](documents/CS-7646.txt) | [URL](https://www.omshub.org/course/CS-7646) |
| 3 | CS-7641 Machine Learning | OMSHub Course Reviews | [File](documents/CS-7641.txt) | [URL](https://www.omshub.org/course/CS-7641) |
| 4 | CS-6515 Intro to Graduate Algorithms | OMSHub Course Reviews | [File](documents/CS-6515.txt) | [URL](https://www.omshub.org/course/CS-6515) |
| 5 | CS-6200 Introduction to Operating Systems | OMSHub Course Reviews | [File](documents/CS-6200.txt) | [URL](https://www.omshub.org/course/CS-6200) |
| 6 | CS-6601 Artificial Intelligence | OMSHub Course Reviews | [File](documents/CS-6601.txt) | [URL](https://www.omshub.org/course/CS-6601) |
| 7 | CS-6750 Human-Computer Interaction | OMSHub Course Reviews | [File](documents/CS-6750.txt) | [URL](https://www.omshub.org/course/CS-6750) |
| 8 | CS-7642 Reinforcement Learning | OMSHub Course Reviews | [File](documents/CS-7642.txt) | [URL](https://www.omshub.org/course/CS-7642) |
| 9 | CS-6476 Computer Vision | OMSHub Course Reviews | [File](documents/CS-6476.txt) | [URL](https://www.omshub.org/course/CS-6476) |
| 10 | ISYE-6501 Intro to Analytics Modeling | OMSHub Course Reviews | [File](documents/ISYE-6501.txt) | [URL](https://www.omshub.org/course/ISYE-6501) |
| 11 | CSE-6040 Computing for Data Analysis | OMSHub Course Reviews | [File](documents/CSE-6040.txt) | [URL](https://www.omshub.org/course/CSE-6040) |
| 12 | CS-7643 Deep Learning | OMSHub Course Reviews | [File](documents/CS-7643.txt) | [URL](https://www.omshub.org/course/CS-7643) |
| 13 | CS-6210 Advanced Operating Systems | OMSHub Course Reviews | [File](documents/CS-6210.txt) | [URL](https://www.omshub.org/course/CS-6210) |
| 14 | MGT-6203 Data Analytics in Business | OMSHub Course Reviews | [File](documents/MGT-6203.txt) | [URL](https://www.omshub.org/course/MGT-6203) |
| 15 | CS-6603 AI, Ethics, and Society | OMSHub Course Reviews | [File](documents/CS-6603.txt) | [URL](https://www.omshub.org/course/CS-6603) |
| 16 | CS-6250 Computer Networks | OMSHub Course Reviews | [File](documents/CS-6250.txt) | [URL](https://www.omshub.org/course/CS-6250) |
| 17 | CS-6400 Database Systems Concepts and Design | OMSHub Course Reviews | [File](documents/CS-6400.txt) | [URL](https://www.omshub.org/course/CS-6400) |
| 18 | CS-6300 Software Development Process | OMSHub Course Reviews | [File](documents/CS-6300.txt) | [URL](https://www.omshub.org/course/CS-6300) |
| 19 | CS-6262 Network Security | OMSHub Course Reviews | [File](documents/CS-6262.txt) |[URL](https://www.omshub.org/course/CS-6262) |
| 20 | CS-6475 Computational Photography | OMSHub Course Reviews | [File](documents/CS-6475.txt) | [URL](https://www.omshub.org/course/CS-6475) |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 200 tokens (approximately 800–900 characters)

**Overlap:** 30 tokens (approximately 120 characters)

**Reasoning:**

Reviews on OMSHub vary dramatically in length -- from under 100 words to over 2,500 words. A fixed strategy that works for short reviews must also handle long ones without degrading retrieval quality. Since the data is already collected and stored as plain text with structured fields, no HTML cleaning is required -- the `Review:` field is extracted directly and passed to the splitter.

The primary constraint driving chunk size is the embedding model: `all-MiniLM-L6-v2` has a hard maximum of 256 tokens. Any input exceeding this is silently truncated, meaning content at the end of a long chunk simply disappears during embedding. A 200-token target gives a safe buffer below that ceiling.

Short reviews (under 200 tokens) are kept as a single chunk -- splitting a 150-word review would produce fragments too small to carry meaning. Long reviews (the 2,500-word outliers) get split into 8–12 chunks automatically. This is intentional: a 2,500-word review typically covers projects, exams, workload, and professor quality in separate paragraphs. Smaller focused chunks mean a query about "exam difficulty" retrieves the exam paragraph specifically, rather than a diluted whole-review embedding that competes poorly against shorter, more focused reviews.

The 30-token overlap prevents meaning loss at boundaries. Reviews often use referential language mid-thought ("but that said, the final project..." loses its antecedent without overlap). 30 tokens is enough to carry that context without duplicating significant content across chunks.

**Implementation:** A custom recursive splitter in plain Python with separators `["\n\n", "\n", ". ", " "]` -- attempts to split on paragraph breaks first, then newlines, then sentence boundaries, then words, avoiding mid-sentence cuts wherever possible. No external libraries beyond the core stack are required for this step.

Token counting is handled using `all-MiniLM-L6-v2`'s own tokenizer, which is already loaded as part of `sentence-transformers`. A `token_len()` helper function is passed as the length function to the splitter:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def token_len(text):
    return len(model.tokenizer.encode(text))
```

This ensures "200 tokens" means exactly 200 tokens as counted by the embedding model itself -- not a character-based approximation. Using the model's own tokenizer also guarantees chunk boundaries align with how the model will actually process the text during embedding.

Every chunk inherits its parent review's metadata (course code, course name, semester, year, difficulty rating, workload hours, overall rating, unique `review_id`, and `chunk_index`) so retrieval results remain attributable to their source review. The `review_id` follows the format `COURSEID_SEMESTER_YEAR_INDEX` (e.g. `CS6250_SPRING_2022_001`) and is generated during parsing -- this allows the citation block to distinguish between multiple reviews from the same course and semester. The `chunk_index` (0, 1, 2...) tracks which chunk of a given review was retrieved, making debugging significantly easier when a long review produces 8-12 chunks.

**Stretch feature -- Chunking Strategy Comparison:**

A second strategy using a custom semantic chunker will be tested against the recursive splitter on the same 5 evaluation questions. The semantic chunker is built directly on top of `sentence-transformers` (already in the stack): each sentence in a review is embedded individually, then cosine similarity between adjacent sentence embeddings is computed -- when similarity drops below a threshold, a chunk boundary is inserted. This produces topically coherent chunks (naturally separating the "projects section" from the "exams section") without any additional libraries. The tradeoff is speed: embedding every sentence during ingestion is significantly slower than the plain-Python recursive splitter, and the similarity threshold requires tuning. Results of both strategies will be reported in the evaluation report.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` (runs locally, no API key required)

**Top-k:** 5 chunks per query

5 chunks balances two competing needs: enough retrieved context to surface the variation and contradictions that naturally exist across student opinions, without overloading the Groq LLM prompt and risking rate limit hits on the free tier. A query about workload for CS-7641 should return perspectives from multiple semesters and student backgrounds, not just the single closest match.

Initial implementation will use top-k=5. During evaluation I will compare k=5 through k=8 on the five evaluation questions to find a k that returns the most accurate responses.

**Production tradeoff reflection:**

`all-MiniLM-L6-v2` is appropriate for this project given the zero-cost constraint, but several tradeoffs would be reconsidered in a production deployment:

- **Context length:** The 256-token limit is a real constraint. Models like `text-embedding-3-small` (OpenAI) or `embed-english-v3.0` (Cohere) support much longer inputs, which would reduce or eliminate the need to split shorter reviews at all.
- **Domain specificity:** All-MiniLM was trained on general web text. A model fine-tuned on academic or student-written text would likely produce better similarity scores for domain-specific vocabulary (e.g., "proctored exam," "project rubric," "office hours").
- **Multilingual support:** Not a concern for this corpus (English-only), but models like `paraphrase-multilingual-MiniLM-L12-v2` would matter for a platform serving international students.
- **Latency vs. accuracy:** Local embedding is fast and free but less accurate than larger API-based models. For a production system with real users, the accuracy improvement from a model like `text-embedding-3-large` may justify the API cost.

**Stretch feature -- Metadata Filtering:**

ChromaDB supports metadata filtering natively. Users will be able to narrow results by: `course_id` (e.g., only CS-7641 reviews), `semester` (Fall/Spring), `year`, `difficulty` (1–5 scale), and `workload_hours`. This makes queries like "What do students say about CS-6515 exams in recent semesters?" significantly more precise than semantic search alone.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | How many hours per week does CS-7641 Machine Learning typically require? | Reviews consistently report 15-20+ hours per week, making it one of the most time-intensive courses in the program. |
| 2 | Is CS-6250 Computer Networks a good course to pair with a more difficult course? | Yes -- multiple reviews describe it as an easy, manageable course that pairs well with heavier workloads, typically requiring around 6-7 hours per week. |
| 3 | What do students say about the projects in CS-6200 Introduction to Operating Systems? | Students report the projects are challenging and time-consuming, often involving systems programming in C, and represent the bulk of the course's difficulty. |
| 4 | How difficult are the exams in CS-6515 Intro to Graduate Algorithms? | Reviews describe the exams as among the hardest in the OMS program -- proctored, algorithm-heavy, and requiring deep conceptual preparation beyond just completing assignments. |
| 5 | Is CS-6603 AI, Ethics, and Society considered an easy course? | Generally yes -- reviews describe it as a lighter workload course focused on discussion and writing rather than heavy programming, suitable for pairing with a harder technical course. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Review length variance breaking chunk coherence.** Reviews range from under 100 words to over 2,500 words. For very long reviews, the recursive splitter will produce 8–12 chunks per review. If a student builds a cumulative argument across multiple paragraphs ("the course starts easy but the final project is where everything gets hard -- and that's actually where I learned the most"), splitting that arc across chunk boundaries will fragment the reasoning. A query about "final project difficulty" might retrieve only the second half of that argument, losing the contrast that gives it meaning. Mitigated by 30-token overlap, but not fully solved -- this is a known failure mode to document in the evaluation report.

2. **Contradictory reviews producing incoherent answers.** For popular courses like CS-7641, the corpus will contain reviews ranging from "this destroyed me" to "manageable if you have ML experience." The LLM must surface this variation rather than averaging it into a falsely confident answer. This will be addressed in the system prompt by instructing the model to represent disagreement explicitly ("some students report... while others note...") and always attribute claims to specific reviews by course and semester. A failure case to test: ask about difficulty for a course with bimodal reviews and check whether the response acknowledges the split or collapses it.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                    (no scraping -- files pre-collected)         │
│                                                                 │
│  Local .txt files  ──►  Custom Python parser  ──►  Review dicts │
│  (documents/ folder, 20 files, approximately 20 reviews each,   │
│   Course / Semester / Workload / Difficulty /                   │
│   Overall / Review fields, separated by ---)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ structured review dicts
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CHUNKING STAGE                            │
│                                                                 │
│  Raw review text  ──►  Custom recursive splitter (plain Python) │
│                        (chunk_size=200 tokens,                  │
│                         chunk_overlap=30 tokens,                │
│                         separators: \n\n → \n → . → space)      │
│                                                                 │
│  Each chunk inherits metadata:                                  │
│  { course_id, course_name, semester, year,                      │
│    difficulty, workload_hrs, overall_rating,                    │
│    review_id, chunk_index }                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ list of (chunk_text, metadata) pairs
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING + VECTOR STORE                       │
│                                                                 │
│  chunk_text  ──►  sentence-transformers                         │
│                   (all-MiniLM-L6-v2, local)                     │
│                ──►  384-dim embedding vector                    │
│                                                                 │
│  (vector + metadata)  ──►  ChromaDB  (local persistent store)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ indexed vector store
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL STAGE                          │
│                                                                 │
│  User query  ──►  all-MiniLM-L6-v2  ──►  query embedding        │
│                                                                 │
│  ChromaDB semantic search  ──►  top-5 chunks                    │
│  + optional metadata filters                                    │
│    (course_id, semester, difficulty, workload)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ top-5 chunks + source metadata
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GENERATION STAGE                           │
│                                                                 │
│  System prompt + retrieved chunks  ──►  Groq API                │
│                                         (llama-3.3-70b)         │
│                                                                 │
│  Response includes:                                             │
│  - Answer grounded in retrieved chunks only                     │
│  - Source attribution block (programmatically appended):        │
│      Sources:                                                   │
│      - CS-6250 (Spring 2022) [CS6250_SPRING_2022_001]           │
│      - CS-6250 (Fall 2022)   [CS6250_FALL_2022_003]             │
│  - Acknowledged disagreement when reviews conflict              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ grounded, cited answer
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY INTERFACE                           │
│                                                                 │
│  Gradio web UI (gradio>=6.9.0)                                  │
│  - Text input for natural language query                        │
│  - Answer display (grounded response with conflict handling)    │
│  - Sources display (course code + semester, programmatically    │
│    appended from chunk metadata)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 -- Ingestion and chunking:**

- **Tool:** Claude
- **Input:** The Documents table above, the Chunking Strategy section, and the exact format of the pre-collected `.txt` files -- each file contains reviews for one course, reviews are separated by `---`, and each review has labeled fields (`Course:`, `Course Name:`, `Semester:`, `Date:`, `Workload:`, `Difficulty:`, `Overall:`, `Review:`)
- **Expected output:** A Python script `ingest.py` implementing (1) a file loader that reads all 20 `.txt` files from `documents/`, (2) a parser that splits each file on `---` and extracts the labeled fields into a structured dict per review -- no HTML cleaning needed since data is already plain text, (3) a normalizer that standardizes metadata fields (e.g. parse `"6 hrs/week"` -> `6`, `"2/5 Easy"` -> `2`) and generates a `review_id` in the format `COURSEID_SEMESTER_YEAR_INDEX` (e.g. `CS6250_SPRING_2022_001`), (4) a custom recursive splitter in plain Python using separators `["\n\n", "\n", ". ", " "]` with `token_len()` as the length function (using `all-MiniLM-L6-v2`'s own tokenizer), chunk_size=200 tokens and overlap=30 tokens, and (5) a function that attaches metadata to each chunk -- including `chunk_index` tracking position within the parent review -- and returns a list of dicts ready for embedding
- **Verification:** Run on 2 `.txt` files from `documents/`, manually inspect 5 chunks per file -- confirm no chunk exceeds 256 tokens, confirm all metadata fields are correctly parsed (including numeric normalization of workload and difficulty), confirm short reviews are not split unnecessarily. Also verify total chunk count falls between 50 and 2,000 across all 20 files -- fewer than 50 suggests chunks are too large; more than 2,000 suggests they are too small.

**Milestone 4 -- Embedding and retrieval:**

- **Tool:** Claude
- **Input:** The Retrieval Approach section, the chunk dict schema from Milestone 3, and the stretch feature description for Metadata Filtering
- **Expected output:** A script `embed_and_store.py` that (1) loads chunks from Milestone 3, (2) embeds them using `sentence-transformers` with `all-MiniLM-L6-v2`, (3) stores vectors + metadata in a local ChromaDB collection, and (4) exposes a `query(text, filters=None, top_k=5)` function that accepts optional metadata filters and returns the top-5 chunks with their source metadata
- **Verification:** Run 3 of the 5 evaluation questions manually, confirm returned chunks are topically relevant, confirm distance scores on top results are below 0.5 (scores above 0.6–0.7 indicate weak matches and require debugging chunk size or cleaning), confirm metadata filters correctly narrow results when applied

**Milestone 5 -- Generation and interface:**

- **Tool:** Claude
- **Input:** The Evaluation Plan section, the retrieval function from Milestone 4, and two explicit requirements: (1) grounding must be enforced in the system prompt -- the LLM must be instructed to answer *only* from the retrieved context and explicitly say "I don't have enough information" when the documents don't cover the question; (2) source attribution must be programmatically appended from chunk metadata, not left to the LLM to add on its own
- **Expected output:** (1) A `generate.py` module that constructs a system prompt enforcing grounded responses and conflict acknowledgment, calls the Groq API with retrieved chunks as context, and programmatically appends a citation block from retrieved chunk metadata in the format `- COURSE_NAME (Semester Year) [review_id]` -- never relying on the LLM to self-attribute; (2) A Gradio `app.py` with a query text box, a response display showing the grounded answer, and a separate sources display listing each retrieved chunk with course, semester, and review_id
- **Verification:** Run all 5 evaluation questions, record retrieved chunks and responses, confirm no response makes a claim unsupported by the retrieved context, confirm sources are cited in every answer, confirm that asking a question not covered by the corpus produces an explicit "I don't have enough information" response rather than a plausible-sounding hallucination

> **Interface choice note:** Gradio is used instead of Streamlit because the milestone instructions provide a Gradio starter skeleton and it requires less boilerplate for this use case.