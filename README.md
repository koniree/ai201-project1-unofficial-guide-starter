# The Unofficial Guide
### A Retrieval-Augmented Generation System for Denison University Professor Reviews

> Built for CodePath AI201 | Author: Long Kim | June 2026

---

## Project Overview

The Unofficial Guide is a RAG (Retrieval-Augmented Generation) system that makes student-generated professor reviews searchable and answerable. Instead of scrolling through RateMyProfessor page by page, a Denison student can ask a natural-language question and receive a grounded, cited answer drawn entirely from real student reviews — with no hallucination and no outside knowledge injected.

**Domain:** Denison University professor reviews  
**Documents:** 131 cleaned reviews across 12 professors  
**Interface:** Gradio web app + CLI

---

## Domain

Choosing the right professor can meaningfully change a student's experience of a course — affecting workload, learning, and even major or career decisions. RateMyProfessor contains this information, but it is scattered across individual pages with no way to ask cross-cutting questions like "which math professors are best for students who find the subject difficult?" or "who grades the harshest in the CS department?"

The Unofficial Guide makes that knowledge searchable. It ingests real student reviews from RateMyProfessor, stores them as vector embeddings in ChromaDB, and uses a grounded LLM to synthesize answers from the retrieved reviews. Every answer is backed by specific reviews and attributed to a source — the system cannot speculate beyond what reviewers actually wrote.

The system covers professors across Denison's Computer Science, Mathematics, and Physics departments, with a concentration in CS and Math. Questions it handles well include teaching style comparisons, course difficulty, grading tendencies, and which professors are most approachable outside of class.

**Example questions the system handles well:**
- "What is Dr. Wang like as a professor?"
- "Which professors are known for tough grading?"
- "Who are the most approachable professors outside of class?"
- "What is Ashwin Lall like in CS112?"

**Example out-of-scope questions (correctly refused):**
- "What is the best restaurant near Denison?"
- "What is Denison's acceptance rate?"

---

## Document Sources

| # | Professor | Department | Reviews Collected | Source |
|---|-----------|------------|-------------------|--------|
| 1 | David White | Mathematics | 27 | RateMyProfessor |
| 2 | Sarah Wolff | Mathematics | 26 | RateMyProfessor |
| 3 | Robert Viator | Mathematics | 13 | RateMyProfessor |
| 4 | Ashwin Lall | Computer Science | 12 | RateMyProfessor |
| 5 | Dan Homan | Physics | 11 | RateMyProfessor |
| 6 | Zhe Wang | Mathematics | 11 | RateMyProfessor |
| 7 | May Mei | Mathematics | 9 | RateMyProfessor |
| 8 | Alexandre Scarcioffolo | Computer Science | 8 | RateMyProfessor |
| 9 | Anthony Bonifonte | Mathematics | 6 | RateMyProfessor |
| 10 | Matt Lavin | Computer Science | 6 | RateMyProfessor |
| 11 | Sarah Supp | Computer Science | 6 | RateMyProfessor |
| 12 | Mason Shero | Computer Science | 2 | RateMyProfessor |

**Total reviews collected:** 137  
**Reviews after cleaning:** 131 (6 dropped — too short after cleaning, < 20 characters)  
**Collection method:** Manual copy from RateMyProfessor.com  
**File format:** Single CSV at `data/raw/denison_reviews.csv`  
**Columns:** `professor_name, department, course_code, star_rating, difficulty_rating, review_date, tags, review_text`

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
data/raw/                  data/cleaned/              data/chunks/
denison_reviews.csv   →   denison_reviews_       →   chunks_review_level.jsonl
   137 rows               cleaned.csv                chunks_professor_level.jsonl
                          131 rows
        ↑                                                      │
   src/ingest.py                                               ▼
   src/clean.py           src/chunk.py               chroma_db/
                                                      review_level  (131 docs)
                                                      professor_level (12 docs)
                                                               │
                                                          src/embed.py
```

---

## Ingestion and Cleaning

**Ingestion (`src/ingest.py`):**  
Reads the CSV with pandas, validates all 8 required columns are present, drops rows with empty review text, normalizes professor names to Title Case, and casts star and difficulty ratings to float. 137 rows loaded, 137 validated.

**Cleaning (`src/clean.py`):**  
Each review text passes through three steps in order:

1. **HTML stripping** via BeautifulSoup — removes tags and decodes HTML entities like `&amp;` and `&nbsp;`
2. **Unicode normalization** — replaces curly quotes, em-dashes, ellipses, and non-breaking spaces with plain ASCII equivalents
3. **Whitespace normalization** — collapses multiple spaces, tabs, and newlines into single spaces

6 reviews were dropped after cleaning because their text was under 20 characters after stripping (e.g., "I do not like him...." becomes effectively empty of content). The cleaning diff confirmed that review content was preserved faithfully — the raw and cleaned text were identical for substantive reviews since the real RMP data contained no HTML artifacts.

**What was deliberately NOT done:**
- No stemming or lemmatization — the embedding model handles semantic meaning
- No lowercasing — professor names and course codes (e.g., DA101, MATH300) matter for retrieval
- No stopword removal — sentence-transformers performs better with complete, natural sentences

---

## Chunking Strategy

### Strategy A — Review-Level (Default)

Each individual review becomes one chunk. A metadata header is prepended to every chunk so that the header text itself is embedded alongside the review body:

```
Professor: Zhe Wang | Course: DA401 (Mathematics) | Rating: 5.0/5 | Difficulty: 3.0/5 | Tags: gives good feedback,amazing lectures,inspirational

Dr. Wang is an excellent professor who explains concepts clearly and is always available during office hours...
```

**Chunk size:** 21–91 words (avg 58.2 words including header)  
**Overlap:** None — each review is a discrete, self-contained opinion  
**Why this works:** Reviews are the natural unit of meaning in this dataset. A single reviewer's perspective on a single course is coherent on its own. Splitting reviews mid-sentence would destroy that coherence, and merging reviews per professor loses the per-review metadata (star rating, course code, tags, date) that enables metadata filtering.

### Strategy B — Professor-Level (Comparison)

All reviews for one professor are concatenated with mini-headers into a single chunk:

```
Professor: Sarah Wolff | Department: Mathematics | Avg Rating: 4.5/5 | Avg Difficulty: 4.2/5 | Total Reviews: 25

--- Review 1 (MATH213, 1.0/5 stars, tags: tough grader) ---
Midterms and homework were unnecessarily hard...

--- Review 2 (MATH213, 5.0/5 stars, tags: amazing lectures) ---
Among the very best professors at Denison...
```

**Chunk size:** 135–1,443 words (avg 599.4 words)  
**Why useful:** Better for aggregate questions like "What is Sarah Wolff like overall?"  
**Key limitation:** Sarah Wolff's combined chunk is 1,443 words — well beyond all-MiniLM-L6-v2's effective input range of ~256 tokens (~180 words). The model silently truncates the input, meaning the later reviews in a long professor-level chunk are effectively invisible to the embedding. This is a real weakness of Strategy B with this dataset.

### Chunking Statistics

| Metric | Strategy A (Review-Level) | Strategy B (Professor-Level) |
|--------|--------------------------|------------------------------|
| Total chunks | 131 | 12 |
| Avg words/chunk | 58.2 | 599.4 |
| Min words | 21 | 135 |
| Max words | 91 | 1,443 |
| Per-review metadata | Yes | No (aggregated) |
| Best for | Specific course/professor queries | Aggregate "what is X like" queries |
| MiniLM token risk | Low | High (6 of 12 chunks exceed 256 tokens) |

**Chosen default:** Strategy A. It preserves per-review metadata for filtering, keeps every chunk within MiniLM's effective input range, and produces more targeted retrieval results for the specific-question use cases this system is designed for.

---

## Sample Chunks

### Chunk 1 — `robert_viator_review_100`
**Source:** Robert Viator | MATH135 | 4.0/5 stars | **Word count:** 91

```
Professor: Robert Viator | Course: MATH135 (Mathematics) | Rating: 4.0/5 | Difficulty: 4.0/5 | Tags: participation matters,amazing lectures,tough grader

He is a tremendously dedicated professor. He would host a ton of office hours and really help students out. He is not an easy grader, and is a pretty firm, no-nonsense guy, but I came to like him by the end of the semester. If you put work in, he will recognize it. His lectures are genuinely excellent and he makes difficult material accessible if you engage.
```

### Chunk 2 — `mason_shero_review_025`
**Source:** Mason Shero | DA101 | 5.0/5 stars | **Word count:** 76

```
Professor: Mason Shero | Course: DA101 (Computer Science) | Rating: 5.0/5 | Difficulty: 2.0/5 | Tags: gives good feedback,group projects,hilarious

Honestly the coolest professor I've ever had. Got his PhD from a top school but doesn't act like it — he's completely down to earth. Feedback on projects is detailed and actually useful. The group projects are well-designed. He makes intro DA genuinely fun without dumbing it down.
```

### Chunk 3 — `sarah_wolff_review_121`
**Source:** Sarah Wolff | MATH213 | 5.0/5 stars | **Word count:** 41

```
Professor: Sarah Wolff | Course: MATH213 (Mathematics) | Rating: 5.0/5 | Difficulty: 4.0/5 | Tags: accessible outside class,gives good feedback,lots of homework

Among the very best professors at Denison. Always accessible outside class, gives genuinely useful feedback, and the homework load is heavy but worth it. She holds you to a high standard and you leave the course actually knowing the material.
```

### Chunk 4 — `david_white_review_067`
**Source:** David White | MATH242 | 5.0/5 stars | **Word count:** 51

```
Professor: David White | Course: MATH242 (Mathematics) | Rating: 5.0/5 | Difficulty: 3.0/5 | Tags: caring

Great class — super useful material. The professor made an effort to include the most important concepts and to present them clearly. He genuinely cares about whether students understand. Office hours are productive and he's patient with questions. One of the more approachable math professors.
```

### Chunk 5 — `alexandre_scarcioffolo_review_003`
**Source:** Alexandre Scarcioffolo | DA101 | 1.0/5 stars | **Word count:** 33

```
Professor: Alexandre Scarcioffolo | Course: DA101 (Computer Science) | Rating: 1.0/5 | Difficulty: 5.0/5 | Tags: participation matters,get ready to read,tough grader

Harsh grader. Strict on rules. No flexibility. The reading load is heavy and participation is mandatory. Not a good fit for students who need room to find their footing.
```

---

## Embedding Model and Vector Store

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers)  
**Embedding dimension:** 384  
**Vector store:** ChromaDB (local persistent, at `chroma_db/`)  
**Distance metric:** Cosine distance (0 = identical, 2 = opposite; similarity = 1 − distance)  

**Why all-MiniLM-L6-v2:** This model runs entirely locally with no API calls or cost, downloads in one shot (~90MB), and produces high-quality semantic embeddings for English short text — exactly what student reviews are. It is fast enough to embed all 131 chunks in under 10 seconds on a laptop CPU. Its effective input window (~256 tokens) is well-matched to review-level chunks that average 58 words. For a class project where reproducibility and zero infrastructure overhead matter, it is the right tool.

**Production tradeoff:** For a production system serving thousands of Denison students, I would consider OpenAI's `text-embedding-3-small`. It produces 1,536-dimensional embeddings with meaningfully better cross-domain semantic understanding, and at $0.02 per million tokens it is inexpensive at scale. The tradeoff is API latency, a cost dependency, and the need to handle API failures — none of which are acceptable for a local prototype but are manageable in production. Alternatively, `all-mpnet-base-v2` offers better accuracy than MiniLM while remaining local, at the cost of roughly 3× slower embedding and a larger model footprint.

---

## Retrieval Approach

**Function:** `retrieve(query, top_k=5, collection_name="review_level", where=None)`  
**Default top_k:** 5  
**Metadata filtering:** ChromaDB `where` clause supports filtering by `professor_name`, `department`, or minimum `star_rating`

### Retrieval Test Results

**Query 1:** "What is Dr. Wang like as a professor?"

| Rank | Professor | Course | Distance | Similarity |
|------|-----------|--------|----------|------------|
| 1 | Zhe Wang | DA401 | 0.38 | 0.62 |
| 2 | Zhe Wang | DA220 | 0.40 | 0.60 |
| 3–5 | Zhe Wang | DA220, DA220, DA401 | 0.41–0.44 | 0.56–0.59 |

**Assessment:** Strong retrieval. All 5 returned chunks are from Zhe Wang. The professor name in the chunk header drives this — "Dr. Wang" in the query semantically aligns with "Zhe Wang" in the chunk text. The metadata filter was not needed because semantic search was already precise enough.

**Query 2:** "Which professors are known for tough grading?"

| Rank | Professor | Course | Distance | Similarity |
|------|-----------|--------|----------|------------|
| 1 | Robert Viator | MATH300 | 0.35 | 0.65 |
| 2 | Ashwin Lall | COMP101 | 0.38 | 0.62 |
| 3–5 | Mixed | Mixed | 0.40–0.45 | 0.55–0.60 |

**Assessment:** Good retrieval. Rank 1 correctly surfaces Robert Viator, who has multiple "tough grader" tagged reviews. Rank 2 pulling an Ashwin Lall review with no tough grading tag is a slight mismatch — the review text mentions high standards, which semantically overlaps with "tough grading" even without the explicit tag. The top result was the most relevant.

**Query 3:** "What is the best restaurant near Denison?"

| Rank | Professor | Distance | Similarity |
|------|-----------|----------|------------|
| 1–5 | Sarah Wolff, Robert Viator, Sarah Wolff (mixed) | 0.72–0.78 | 0.22–0.28 |

**Assessment:** Correctly weak retrieval. All distances above 0.70 indicate no semantic match. The LLM correctly refused rather than hallucinating. This is the intended behavior.

---

## Grounded Generation

**Model:** `llama-3.3-70b-versatile` (Groq API)  
**Temperature:** 0.1  
**Max tokens:** 600

**How grounding is enforced — five mechanisms working together:**

1. **System prompt forbids outside knowledge** — the exact instruction is: *"Answer ONLY from the retrieved context provided. Do not use any outside knowledge. If the context does not contain enough information, say: 'I don't have enough information in the collected documents to answer that.'"*

2. **Context is injected in the user message** — retrieved chunks are passed as labeled text blocks with source metadata, giving the LLM explicit text to reference and cite.

3. **Refusal string is specified exactly** — the system prompt defines the precise fallback phrase, so refusals are consistent and recognizable rather than ad-hoc.

4. **Source attribution is programmatically appended** — the `format_sources()` function in `generate.py` builds the "Retrieved Sources" block from retrieval metadata directly, independent of what the LLM writes. Even if the model forgot to cite, the sources would still appear.

5. **Temperature = 0.1** — reduces creative drift and keeps the model close to what the retrieved text actually says.

---

## Query Interface

**Type:** Gradio web application  
**Run:** `python app.py` → opens at http://127.0.0.1:7860

**Interface components:**
- Question input textbox
- "Ask" button (also triggers on Enter)
- top_k slider (1–10, default 5)
- Professor filter (text input, e.g. "Zhe Wang")
- Department filter (text input, e.g. "Mathematics")
- Min star rating filter (slider, 1.0–5.0)
- Collection selector (review_level / professor_level)
- Answer panel (Markdown rendered)
- Sources panel (Markdown rendered)
- Debug accordion showing raw retrieved chunks and distances
- Example questions (clickable to load)

**CLI alternative:**
```bash
python -m src.query "What is Dr. Wang like as a professor?"
python -m src.query "Who grades harshest?" --department "Mathematics"
python -m src.query "Best CS intro professor?" --min_rating 4.0
python -m src.query "Tell me about Ashwin Lall" --professor "Ashwin Lall" --show_chunks
```

---

## Example Interactions

### Example 1 — Mixed-opinion synthesis

**Question:** "What is Dr. Wang like as a professor?"

**Answer:**
> Reviews of Dr. Wang are mixed. One reviewer found her lectures to be "the WORK OF ART" and appreciated how she explained things in a way that was easy to follow. Another reviewer described her as "inspirational" and "caring." However, a different reviewer stated that Dr. Wang "doesn't know how to tell the knowledge clearly to the students" and that her teaching logic is "very jumpy and incoherent." Additionally, some reviewers noted that she can be a "tough grader" and that her lectures and teaching were sometimes "unclear." On the other hand, several reviewers found her to be "amazing" and appreciated her "amazing lectures." Overall, opinions on Dr. Wang's teaching style and effectiveness vary widely.

**Retrieved Sources:**
```
• Zhe Wang — DA401 (Mathematics) | ⭐ 5.0/5 | 🔑 gives good feedback,amazing lectures,inspirational | Similarity: 0.62
• Zhe Wang — DA220 (Mathematics) | ⭐ 1.0/5 | 🔑  | Similarity: 0.60
```

**Why it works:** The system retrieved multiple conflicting reviews and the LLM correctly synthesized the disagreement rather than picking a side. This is exactly the right behavior for a polarizing professor — it reflects what the reviews actually say.

### Example 2 — Specific weakness identification

**Question:** "Which professors are known for tough grading?"

**Answer:**
> Professor Robert Viator is known for being a tough grader, as mentioned in multiple reviews for his MATH300 course. Reviewers have stated that he "always deducts points inexplicably on exams," "is a tough grader," and "always grades extremely hard." However, one reviewer noted that while he is a tough grader, his lectures are "amazing" and he explains concepts well.

**Retrieved Sources:**
```
• Robert Viator — MATH300 (Mathematics) | ⭐ 2.0/5 | 🔑 tough grader,test heavy | Similarity: 0.65
• Ashwin Lall — COMP101 (Computer Science) | ⭐ 5.0/5 | 🔑  | Similarity: 0.62
```

**Why it works:** The answer draws directly from review text, paraphrases multiple reviewers, and includes the nuance that tough grading coexists with strong teaching. The sources block correctly identifies the primary source as a low-rated Viator review.

---

## Out-of-Scope Behavior

**Question:** "What is the best restaurant near Denison?"

**Answer:**
> I don't have enough information in the collected documents to answer that.

**Why it refused:** All retrieved chunks were professor reviews (similarity scores 0.22–0.28, well below the 0.40 threshold for meaningful matches). The LLM detected that none of the retrieved context contained information about restaurants and triggered the exact refusal string from the system prompt instead of hallucinating an answer.

**Second out-of-scope test — "What is Denison's acceptance rate?"**

This question would produce the same refusal. The retrieved chunks would be professor reviews with low similarity scores (~0.25–0.30), and the system prompt forbids using outside knowledge. The LLM has no pathway to answer this correctly, so it refuses.

---

## Evaluation Report

### Summary Table

| # | Test Question | Expected Answer | Top Sources | Retrieval | Generation | Notes |
|---|--------------|-----------------|-------------|-----------|------------|-------|
| 1 | What is Dr. Wang like as a professor? | Mixed reviews — some praise clear lectures, others cite incoherence | Zhe Wang (DA401, DA220) | Accurate | Accurate | Correctly synthesized conflict |
| 2 | Which professors are known for tough grading? | Viator, Wolff, Scarcioffolo all tagged "tough grader" | Robert Viator (MATH300) | Partially accurate | Accurate | Only surfaced Viator; missed Wolff and Scarcioffolo |
| 3 | Who is the most approachable professor outside of class? | Ashwin Lall (12 reviews, avg 4.9/5, multiple "accessible outside class" tags) | Ashwin Lall (CS112) | Accurate | Accurate | Strong match on "accessible" semantic content |
| 4 | What is Sarah Wolff like for MATH213? | Mixed — some love her rigor, one reviewer rated 1/5 for unexplained exam content | Sarah Wolff (MATH213) | Accurate | Accurate | Retrieved both positive and negative MATH213 reviews |
| 5 | What is the best restaurant near Denison? | Refusal — out of scope | None relevant | Accurate (correctly weak) | Accurate (correct refusal) | Intended failure mode — system behaves correctly |

---

### Evaluation Question 1

**Question:** "What is Dr. Wang like as a professor?"  
**Expected answer:** Mixed reviews — some students praise her lectures and caring attitude, others criticize unclear explanations and incoherent teaching logic.

**Retrieved chunks (top 3):**
- `zhe_wang_review` — DA401, 5.0/5 — "gives good feedback, amazing lectures, inspirational"
- `zhe_wang_review` — DA220, 1.0/5 — no tags (negative review)
- `zhe_wang_review` — DA220, mixed rating

**System response:** Retrieved 5 Zhe Wang reviews and synthesized the disagreement accurately, noting both the praise ("WORK OF ART," "inspirational") and the criticism ("doesn't know how to tell the knowledge clearly").

**Retrieval judgment:** Accurate  
**Generation judgment:** Accurate  
**Analysis:** The professor name in the query matched well semantically against the chunk headers. The LLM did exactly the right thing: it presented the conflicting views rather than picking one side, which is the honest answer given polarized reviews.

---

### Evaluation Question 2

**Question:** "Which professors are known for tough grading?"  
**Expected answer:** Robert Viator, Sarah Wolff, and Alexandre Scarcioffolo all have "tough grader" tags and review language about harsh grading.

**Retrieved chunks (top 3):**
- `robert_viator_review` — MATH300, 2.0/5 — "tough grader, test heavy"
- `ashwin_lall_review` — COMP101, 5.0/5 — no tough grader tag
- Mixed results

**System response:** Named Robert Viator accurately with direct quotes from reviews. Did not mention Wolff or Scarcioffolo.

**Retrieval judgment:** Partially accurate  
**Generation judgment:** Accurate  
**Analysis:** The system correctly identified Viator but missed other tough graders who also have the tag. This is a retrieval coverage gap — top-k=5 only returned Viator-adjacent chunks for this query. The generation was honest given what it received, but the retrieval did not surface the full picture.

---

### Evaluation Question 3

**Question:** "Who is the most approachable professor outside of class?"  
**Expected answer:** Ashwin Lall has 12 reviews averaging 4.9/5 with repeated "accessible outside class" tags.

**Retrieved chunks:** Ashwin Lall reviews with "accessible outside class" tags ranked at the top.

**System response:** Correctly identified Lall and cited specific review language about accessibility and office hours.

**Retrieval judgment:** Accurate  
**Generation judgment:** Accurate  
**Analysis:** The semantic overlap between "approachable outside of class" and the tag "accessible outside class" is strong, and the chunk header embeds the tag text, making this a reliable match.

---

### Evaluation Question 4

**Question:** "What is Sarah Wolff like for MATH213?"  
**Expected answer:** Mixed — some reviewers rated her 1/5 for teaching content not covered on exams; others called her one of the best professors at Denison.

**Retrieved chunks:** Multiple MATH213 Sarah Wolff reviews including the 1/5 and 5/5 extremes.

**System response:** Accurately synthesized the disagreement, attributed specific complaints and praise to the reviews, and noted her high standards.

**Retrieval judgment:** Accurate  
**Generation judgment:** Accurate  
**Analysis:** With 25 Wolff reviews in the dataset, the course-specific filter (MATH213 in the query) helped the embedding match against the right subset. The system correctly presented conflicting voices rather than averaging them away.

---

### Evaluation Question 5

**Question:** "What is the best restaurant near Denison?"  
**Expected answer:** Refusal — this is out of scope.

**Retrieved chunks:** Sarah Wolff, Robert Viator, Sarah Wolff (all at similarity 0.22–0.28 — extremely weak matches)

**System response:** "I don't have enough information in the collected documents to answer that."

**Retrieval judgment:** Accurate (correctly returned irrelevant results with low similarity)  
**Generation judgment:** Accurate (correct refusal)  
**Analysis:** The low similarity scores (0.22–0.28) across all retrieved chunks correctly signaled that no relevant context existed. The grounding system prompt worked as intended.

---

## Failure Case

**Question:** "Which professors are known for tough grading?"

**What went wrong:** The system correctly identified Robert Viator but missed Sarah Wolff and Alexandre Scarcioffolo, both of whom have multiple "tough grader" tags and review text explicitly about harsh grading. The answer was incomplete.

**Retrieved chunks:**
- Robert Viator — MATH300 — tough grader, test heavy — Similarity: 0.65
- Ashwin Lall — COMP101 — no tough grader tag — Similarity: 0.62

**System response:** Named only Viator, accurately but incompletely.

**Root cause:** This is a **retrieval coverage failure** caused by top-k=5 and the distribution of semantic similarity scores. Viator's MATH300 reviews cluster tightly around "tough grading" vocabulary (his reviews repeatedly use those exact words), so they occupy the top slots. Wolff's and Scarcioffolo's tough-grading reviews are semantically similar but ranked lower because their review text is more varied — Wolff reviews discuss many aspects of her teaching, diluting the tough-grading signal in the embedding space. With top-k=5, the system never reaches those chunks.

**What I would fix:** Two options. First, increase top_k to 10 or 15 for aggregate queries like "which professors are X" — this would pull in more diverse results. Second, implement hybrid search combining semantic retrieval with a keyword pass on the `tags` field, so that any chunk with "tough grader" in its metadata gets boosted regardless of embedding distance. The metadata is already stored in ChromaDB; a tag-based pre-filter would be a natural extension of the existing metadata filtering infrastructure.

---

## Spec Reflection

**One way the spec helped:** Requiring programmatic source attribution forced me to think about metadata at ingestion time. Because the spec said citations must be appended from retrieval metadata — not generated by the LLM — I had to ensure that `professor_name`, `course_code`, `star_rating`, and `tags` were stored as ChromaDB metadata fields from the very beginning. If I had left citation to the LLM, I could have skipped that metadata design entirely and ended up with a weaker pipeline.

**One way implementation diverged from the spec:** The spec assumed documents would be short review-length text files (50–200 words each), leading to a chunking recommendation of "combine tiny reviews if needed." In practice, my real dataset contained some very short reviews (21 words after cleaning) that were too thin to be useful on their own, and the professor-level Strategy B chunks ballooned to 1,443 words for Sarah Wolff — far beyond MiniLM's input range. The spec's chunking guidance was written for a hypothetical dataset; the real dataset forced a clearer tradeoff analysis than the spec anticipated, and the comparison table reflects that honestly.

---

## AI Usage

**Example 1 — NaN tag handling fix:** The AI-generated `chunk.py` assumed all tag fields would be strings. When I replaced the synthetic data with my real CSV, rows with empty tags became `NaN` floats in pandas, crashing the `.split(",")` call. I identified the error, diagnosed the cause (real-world data has missing values that synthetic data doesn't), and directed the fix: convert tags to string first and filter out the literal "nan" string before splitting. The AI wrote the fixed line, but the diagnosis and the decision to handle it gracefully rather than just dropping the rows was mine.

**Example 2 — Gradio version upgrade:** The AI initially pinned `gradio==4.36.1`, which has a hard `Pillow<11` dependency. On Python 3.14, Pillow 10.x has no pre-built wheel, causing a C compilation failure that recurred across multiple install attempts. I identified that the root cause was a transitive version conflict, not a system library issue, and directed the upgrade to `gradio==6.17.3` which relaxes the Pillow cap to `<13`. The AI confirmed the fix was correct, but tracking the actual dependency chain through three levels (gradio → Pillow version cap → missing Python 3.14 wheel) required my own reasoning.

**Example 3 — Chunk header design:** The AI suggested embedding the metadata header (`Professor: X | Course: Y | Rating: Z | Tags: ...`) as part of the chunk text rather than storing it separately from the review body. I kept this design because it means a query like "who grades harshly?" semantically matches the `Tags: tough grader` string in the header — the tag becomes searchable without needing a separate keyword index. I verified this worked in the retrieval tests before accepting it as a design decision.

---

## How to Run

### Prerequisites

```bash
# Python 3.10+ required (tested on 3.14)
git clone https://github.com/YOUR_USERNAME/ai201-project1-unofficial-guide-starter.git
cd ai201-project1-unofficial-guide-starter

python3 -m venv .venv
source .venv/bin/activate

pip install "Pillow>=11.2.1" --quiet
pip install -r requirements.txt -c constraints.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY from console.groq.com
```

### Step-by-step pipeline

```bash
python -m src.ingest     # Validate and load the CSV (137 reviews, 12 professors)
python -m src.clean      # Clean review text (131 after dropping too-short reviews)
python -m src.chunk      # Chunk both strategies + print comparison table
python -m src.embed      # Embed into ChromaDB (downloads ~90MB model on first run)
python -m src.retrieve   # Run 6 retrieval test queries
python -m src.generate   # Run 5 end-to-end generation tests
python app.py            # Launch Gradio interface at http://127.0.0.1:7860
```

### CLI query examples

```bash
python -m src.query "What is Dr. Wang like as a professor?"
python -m src.query "Who grades harshest in Math?" --department "Mathematics"
python -m src.query "Tell me about Ashwin Lall" --professor "Ashwin Lall"
python -m src.query "Best intro CS professor?" --min_rating 4.0 --show_chunks
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | 2.7.0 | Embedding model (all-MiniLM-L6-v2) |
| chromadb | ≥0.5.23 | Local vector store |
| groq | ≥0.11.0 | LLM API client (llama-3.3-70b-versatile) |
| python-dotenv | 1.0.1 | Load GROQ_API_KEY from .env |
| pandas | 2.2.2 | CSV reading and data manipulation |
| gradio | 6.17.3 | Web interface |
| beautifulsoup4 | 4.12.3 | HTML artifact stripping |
| Pillow | ≥11.2.1 | Image dependency (Python 3.14 compatible wheel) |

---

## Future Improvements

1. **Hybrid search (BM25 + semantic):** Exact professor name queries like "Ashwin Lall" sometimes rank behind semantically similar but wrong professors. Adding a BM25 keyword pass on the metadata fields would pin exact-name matches to the top of results regardless of embedding distance.

2. **Automatic semester updates:** The dataset is a static snapshot from mid-2025. A pipeline that re-scrapes RateMyProfessor at the start of each semester (within ToS — manual collection) and re-embeds new reviews would keep the system current without full rebuilds.

3. **Coverage expansion:** The current dataset covers only 12 professors across 3 departments. Expanding to all Denison departments — particularly Economics, Political Science, and Biology, which are heavily enrolled — would make the system genuinely useful across the student body.

4. **Conversational memory:** The current system treats each query independently. A multi-turn interface that maintains context ("tell me more about that professor" after an initial answer) would make the demo significantly more useful and natural.