# The Unofficial Guide
### A Retrieval-Augmented Generation System for Denison University Professor Reviews

> Built for CodePath AI201 | Author: Long Nguyen | [FILL IN: Date]

---

## Project Overview

The Unofficial Guide is a RAG (Retrieval-Augmented Generation) system that makes
student-generated professor reviews searchable and answerable. Instead of scrolling
through RateMyProfessor, a student can ask natural-language questions and receive
grounded, cited answers drawn entirely from real student reviews.

**Domain:** Denison University professor reviews  
**Documents:** [FILL IN: N] individual reviews across [FILL IN: N] professors  
**Interface:** Gradio web app + CLI  

---

## Domain

[FILL IN: 2–3 paragraphs explaining the domain, why it matters, and what
kinds of questions the system can answer.]

**Example questions the system handles well:**
- "Which CS professor is best for students new to programming?"
- "How hard is Dr. Sharma's chemistry class?"
- "Who are the most inspiring professors at Denison?"
- "What professors are known for tough grading?"

**Example out-of-scope questions (correctly refused):**
- "What restaurants are near Denison?"
- "What is the acceptance rate at Denison?"

---

## Document Sources

| # | Professor | Department | Reviews Collected | Source |
|---|-----------|------------|-------------------|--------|
| 1 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 2 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 3 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 4 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 5 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 6 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 7 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 8 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 9 | [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |
| 10| [FILL IN] | [FILL IN]  | [N]               | RateMyProfessor |

**Total reviews:** [FILL IN]  
**Collection method:** Manual copy from RateMyProfessor.com  
**File format:** Single CSV: `data/raw/denison_reviews.csv`

---

## Architecture

```
User Question
      │
      ▼
┌──────────────┐
│  Gradio UI   │  ← app.py
│  (or CLI)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Retrieve   │  ← src/retrieve.py
│  (ChromaDB   │     all-MiniLM-L6-v2 query encoding
│  + metadata  │     cosine distance search
│   filters)   │     optional metadata pre-filter
└──────┬───────┘
       │  top-k chunks + metadata
       ▼
┌──────────────┐
│   Generate   │  ← src/generate.py
│  (Groq LLM   │     llama-3.3-70b-versatile
│  grounded    │     system prompt enforces grounding
│  generation) │     programmatic source attribution
└──────┬───────┘
       │
       ▼
  Answer + Sources
```

---

## Pipeline Diagram

```
data/raw/                data/cleaned/           data/chunks/
denison_reviews.csv  →  denison_reviews_  →  chunks_review_level.jsonl
                        cleaned.csv          chunks_professor_level.jsonl
        ↑                                              │
   src/ingest.py                                       ▼
   src/clean.py         src/chunk.py            chroma_db/
                                                review_level (collection)
                                                professor_level (collection)
                                                       │
                                                  src/embed.py
```

---

## Ingestion and Cleaning

**Ingestion (`src/ingest.py`):**  
Loads the CSV, validates all required columns, drops empty reviews,
normalizes professor names to Title Case, and casts numeric fields.

**Cleaning (`src/clean.py`):**  
Each review text is processed through:
1. HTML stripping (BeautifulSoup) — handles `&amp;`, `<br>`, etc.
2. Unicode normalization — replaces curly quotes, em-dashes with ASCII
3. Whitespace normalization — collapses multiple spaces/newlines

**What was NOT done deliberately:**
- No stemming or lemmatization (embedding model handles token meaning)
- No lowercasing (professor names and course codes matter)
- No stopword removal (sentence-transformers works better with full sentences)

---

## Chunking Strategy

### Strategy A — Review-Level (Default)

Each individual review becomes one chunk. A metadata header is prepended:

```
Professor: Dr. Sarah Chen | Course: CS 181 (Computer Science) | Rating: 4.5/5 | Difficulty: 3.2/5 | Tags: caring,clear,inspirational

Dr. Chen is one of the best professors I've had at Denison...
```

**Chunk size:** ~80–200 words (including header)  
**Overlap:** None — each review is an independent unit of opinion  
**Why:** Reviews are discrete opinions. Splitting them would destroy
the coherence of a single reviewer's perspective. Combining them per
professor is Strategy B (below).

### Strategy B — Professor-Level (Comparison)

All reviews for one professor are concatenated with mini-headers:

```
Professor: Dr. Sarah Chen | Department: Computer Science | Avg Rating: 4.4/5 | ...

--- Review 1 (CS 181, 4.5/5 stars, tags: caring,clear) ---
Dr. Chen is one of the best...

--- Review 2 (CS 271, 5.0/5 stars, tags: amazing lectures) ---
Data Structures with Dr. Chen was...
```

**Chunk size:** ~400–1200 words  
**Why useful:** Better for "What is Dr. Chen like overall?" type questions  
**Limitation:** Loses per-review metadata; may exceed MiniLM's ideal input length

### Chunking Statistics

[FILL IN AFTER RUNNING python -m src.chunk]

| Metric | Strategy A | Strategy B |
|--------|-----------|-----------|
| Total chunks | [N] | [N] |
| Avg words/chunk | [N] | [N] |
| Min words | [N] | [N] |
| Max words | [N] | [N] |
| Per-review metadata | Yes | No (aggregated) |
| Best for | Specific queries | Aggregate queries |

**Chosen default:** Strategy A — preserves metadata for filtering and
keeps chunks within all-MiniLM-L6-v2's effective input range (~256 tokens).

---

## Sample Chunks

### Chunk 1 — [FILL IN: chunk_id]
**Source:** [professor name] | [course] | [rating]/5 stars  
**Strategy:** Review-Level  
**Word count:** [N]

```
[FILL IN: Paste actual chunk text here after running python -m src.chunk]
```

### Chunk 2 — [FILL IN: chunk_id]
**Source:** [professor name] | [course] | [rating]/5 stars  
**Word count:** [N]

```
[FILL IN]
```

### Chunk 3 — [FILL IN: chunk_id]
**Source:** [professor name] | [course] | [rating]/5 stars  
**Word count:** [N]

```
[FILL IN]
```

### Chunk 4 — [FILL IN: chunk_id]
**Source:** [professor name] | [course] | [rating]/5 stars  
**Word count:** [N]

```
[FILL IN]
```

### Chunk 5 — [FILL IN: chunk_id]
**Source:** [professor name] | [course] | [rating]/5 stars  
**Word count:** [N]

```
[FILL IN]
```

---

## Embedding Model and Vector Store

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers)  
**Embedding dimension:** 384  
**Vector store:** ChromaDB (local persistent)  
**Distance metric:** Cosine distance  

**Why all-MiniLM-L6-v2:**  
[FILL IN: In your own words — speed, quality, local, no API cost, good for English short text]

**Production tradeoff:**  
For a production system serving thousands of Denison students, I would consider
[FILL IN: e.g., text-embedding-3-small from OpenAI, or all-mpnet-base-v2 for
higher accuracy] because [FILL IN: reasoning].
The tradeoff is [FILL IN: cost/speed/accuracy comparison].

---

## Retrieval Approach

**Function:** `retrieve(query, top_k=5, collection_name="review_level", where=None)`  
**Default top_k:** 5  
**Metadata filtering:** ChromaDB `where` clause (professor, department, min rating)

### Retrieval Test Results

**Query 1:** "Which CS professors are good at explaining concepts clearly?"  
[FILL IN AFTER RUNNING python -m src.retrieve]
- Rank 1: [professor] | Distance: [N] | Preview: [...]
- Rank 2: [professor] | Distance: [N] | Preview: [...]
- Rank 3: [professor] | Distance: [N] | Preview: [...]
- **Assessment:** [Did the right chunks come back? Were they relevant?]

**Query 2:** "How hard is Dr. Sarah Chen's courses?" (with professor filter)  
[FILL IN]
- Rank 1: [chunk] | Distance: [N]
- **Assessment:** [Did filtering work correctly?]

**Query 3:** "What professors are known for tough grading?"  
[FILL IN]
- **Assessment:** [Did reviews with "tough grader" tag surface?]

---

## Grounded Generation

**Model:** `llama-3.3-70b-versatile` (Groq API)  
**Temperature:** 0.1  
**Max tokens:** 600  

**How grounding is enforced:**

1. **System prompt** explicitly forbids outside knowledge:  
   *"Answer ONLY from the retrieved context provided. Do not use any outside knowledge."*

2. **Context injection:** Retrieved chunks are passed in the user message,
   clearly labeled with source metadata the LLM can reference.

3. **Refusal instruction:** The system prompt specifies the exact refusal
   string: *"I don't have enough information in the collected documents to answer that."*

4. **Programmatic citation:** Source attribution is built from retrieval
   metadata and appended to every response — it does not depend on the LLM.

5. **Low temperature:** 0.1 reduces the model's tendency to add creative
   content beyond what the context contains.

---

## Query Interface

**Type:** Gradio web application  
**Run:** `python app.py` → opens at http://127.0.0.1:7860

**Interface components:**
- Question input textbox
- "Ask" button (also triggers on Enter)
- top_k slider (1–10, default 5)
- Professor filter (text input)
- Department filter (text input)
- Min star rating filter (slider)
- Collection selector (review_level / professor_level)
- Answer panel (Markdown)
- Sources panel (Markdown)
- Debug accordion (raw chunks)
- Example questions (click to load)

**CLI alternative:**  
```bash
python -m src.query "Which CS professor is best for beginners?"
python -m src.query "How hard is Dr. Sharma?" --professor "Dr. Priya Sharma"
python -m src.query "Best bio course?" --department "Biology" --top_k 3
python -m src.query "Good intro courses?" --min_rating 4.5 --show_chunks
```

---

## Example Interactions

### Example 1 — Strong Grounded Answer

**Question:** "What do students think about Dr. Sarah Chen's teaching style?"

**Answer:**  
[FILL IN: Paste real system output here after running the pipeline]

**Retrieved Sources:**  
[FILL IN]

**Why it works:** [1–2 sentences]

---

### Example 2 — Aggregate Recommendation

**Question:** "Who are the most inspiring professors at Denison?"

**Answer:**  
[FILL IN]

**Retrieved Sources:**  
[FILL IN]

---

## Out-of-Scope Behavior

**Question:** "What restaurants are near Denison University?"

**Answer:**  
"I don't have enough information in the collected documents to answer that."

**Why it refused:** The retrieved chunks were review text about professors.
None contained information about restaurants. The LLM correctly followed
the grounding instruction and refused rather than hallucinating.

[FILL IN: Add 1–2 more real out-of-scope examples with actual retrieved chunk distances]

---

## Evaluation Report

### Summary Table

| # | Test Question | Expected Answer | Top Sources | Retrieval | Generation | Notes |
|---|--------------|-----------------|-------------|-----------|------------|-------|
| 1 | [FILL IN] | [FILL IN] | [FILL IN] | [accurate/partial/inaccurate] | [accurate/partial/inaccurate] | [FILL IN] |
| 2 | [FILL IN] | [FILL IN] | [FILL IN] | | | |
| 3 | [FILL IN] | [FILL IN] | [FILL IN] | | | |
| 4 | [FILL IN] | [FILL IN] | [FILL IN] | | | |
| 5 | [FILL IN] | [FILL IN] | [FILL IN] | | | |

---

### Evaluation Question 1

**Question:** [FILL IN]  
**Expected answer:** [FILL IN]  

**Retrieved chunks:**  
[FILL IN: list the top 3 chunks with source names and brief text]

**System response:**  
[FILL IN: paste actual output]

**Retrieval judgment:** [accurate / partially accurate / inaccurate]  
**Generation judgment:** [accurate / partially accurate / inaccurate]  
**Analysis:** [1–3 sentences: what worked, what didn't, why]

---

### Evaluation Question 2

[FILL IN — same format]

---

### Evaluation Question 3

[FILL IN — same format]

---

### Evaluation Question 4

[FILL IN — same format]

---

### Evaluation Question 5

[FILL IN — same format — this should be your failure or weakness case]

---

## Failure Case

**Question:** [FILL IN: the question that exposed a weakness]

**What went wrong:**  
[FILL IN: Be specific. Did retrieval fail? Did it retrieve the wrong chunks?
Did the LLM go off-script? Was the coverage too thin? Name the specific
pipeline component that failed and why.]

**Example of the failure:**  
Retrieved chunks: [list them]  
System response: [paste it]

**Root cause:**  
[FILL IN: One of these, with explanation:
- "The query used vocabulary not present in the reviews" (retrieval miss)
- "Only N reviews existed for this professor, none mentioning X" (thin coverage)
- "The LLM inferred beyond the context despite the system prompt" (grounding failure)
- "Strategy A's short chunks missed the aggregate picture" (chunking limitation)]

**What I would fix:**  
[FILL IN: Concrete fix — more reviews, different chunking, hybrid search, etc.]

---

## Spec Reflection

**One way the spec helped:**  
[FILL IN: A specific requirement that made your project better. E.g., "Requiring
programmatic citation forced me to think about metadata at ingestion time..."]

**One way implementation diverged from the spec:**  
[FILL IN: Something you did differently from what was originally planned, and
a genuine reason why. Not "I ran out of time" — an actual design insight.]

---

## AI Usage

> The assignment requires honesty about AI assistance. Here are specific examples
> of where AI helped and what I changed or overrode.

**Example 1:**  
AI generated the initial cleaning functions in `src/clean.py`.
I [FILL IN: describe what you changed — did you add a cleaning step AI missed?
Did you remove something that was overkill? Did you change the order?]

**Example 2:**  
AI suggested the chunk header format (Professor | Course | Rating | Tags).
I [FILL IN: did you adjust the format? Add or remove a field? Decide on a
different separator? Explain your reasoning.]

[FILL IN: Add 1–2 more specific examples — the assignment asks for at least 2]

---

## How to Run

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Create .env from template
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Step-by-step pipeline

```bash
# 1. Validate and load the CSV
python -m src.ingest

# 2. Clean review text
python -m src.clean

# 3. Chunk documents (both strategies + comparison)
python -m src.chunk

# 4. Embed chunks into ChromaDB
python -m src.embed

# 5. Test retrieval
python -m src.retrieve

# 6. Test end-to-end generation
python -m src.generate

# 7. Ask a single question via CLI
python -m src.query "Which CS professor is best for beginners?"

# 8. Launch the Gradio interface
python app.py
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | 2.7.0 | Embedding model |
| chromadb | 0.5.0 | Vector store |
| groq | 0.9.0 | LLM API client |
| python-dotenv | 1.0.1 | .env loading |
| pandas | 2.2.2 | CSV handling |
| gradio | 4.36.1 | Web interface |
| beautifulsoup4 | 4.12.3 | HTML cleaning |

---

## Future Improvements

1. [FILL IN: e.g., Automatic semester updates — re-collect reviews each term]
2. [FILL IN: e.g., Hybrid search (BM25 + semantic) for exact name queries]
3. [FILL IN: e.g., Conversational memory for follow-up questions]
4. [FILL IN: e.g., Coverage expansion to more departments]
