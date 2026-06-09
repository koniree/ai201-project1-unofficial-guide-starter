# planning.md — The Unofficial Guide
## CodePath AI201 Project Specification

---

## 1. Domain

**Domain:** Denison University Professor Reviews

**Why this knowledge matters:**
Denison students often rely on word-of-mouth or generic RateMyProfessor pages to choose professors, but those pages aren't searchable by question — you can't ask "who is the most accessible CS professor?" and get a synthesized answer. This RAG system lets students ask natural questions and get grounded answers drawn from real peer reviews, surfacing patterns across multiple reviewers that would be tedious to read manually. It solves the "scroll through 20 reviews" problem with a direct, cited answer.

**Target users:**
Primarily incoming or returning Denison students choosing courses during registration. Secondarily, first-year students who don't yet have a peer network to ask for professor recommendations. The system is especially useful for students picking between multiple sections of the same course or trying to understand workload expectations before enrolling.

---

## 2. Document Sources

**Source:** RateMyProfessor.com — manually collected reviews for Denison professors.

**Number of documents:** 12 professors, 131 individual reviews.

**Professors included:**

- Sarah Wolff — Mathematics (25 reviews, avg 4.5★)
- David White — Mathematics (24 reviews, avg 4.1★)
- Robert Viator — Mathematics (13 reviews, avg 3.2★)
- Ashwin Lall — Computer Science (12 reviews, avg 4.9★)
- Dan Homan — Physics (11 reviews, avg 4.7★)
- Zhe Wang — Mathematics (11 reviews, avg 3.0★)
- May Mei — Mathematics (9 reviews, avg 2.8★)
- Alexandre Scarcioffolo — Computer Science (7 reviews, avg 4.0★)
- Anthony Bonifonte — Mathematics (6 reviews, avg 3.2★)
- Matt Lavin — Computer Science (6 reviews, avg 3.7★)
- Sarah Supp — Computer Science (5 reviews, avg 4.6★)
- Mason Shero — Computer Science (2 reviews, avg 5.0★)

**Why these professors?**
The focus was on the three departments with the most student traffic at Denison: Mathematics, Computer Science, and Physics. These are high-enrollment departments where students frequently need to choose between professors or sections. Professors with at least 5 reviews were prioritized so the system has enough signal per professor to synthesize meaningful answers rather than returning a single sparse review. The two Computer Science and Physics professors with fewer reviews (Shero, Supp, Homan) were included because they cover high-demand courses like DA101 and ASTR100.

**Data collection method:**
Manually copied from RateMyProfessor. Stored as a single CSV file:
`data/raw/denison_reviews.csv` with columns:
`professor_name, department, course_code, star_rating, difficulty_rating, review_date, tags, review_text`

Reviews span **April 2023 – September 2024**.

---

## 3. Chunking Strategy

### Strategy A — Review-Level (CHOSEN DEFAULT)

**Chunk size:** One review per chunk (avg 58 words, min 21, max 91 words)
**Overlap:** None (reviews are discrete, independent units)
**Rationale:** Each review is a complete, self-contained opinion from a single student about a specific course experience. Splitting at the review boundary preserves the full sentiment and context of that opinion — a review that praises a professor's feedback style means something different if it's from a 300-level course vs. an intro course. Keeping reviews intact also preserves per-review metadata (star rating, difficulty, tags, course code) so retrieval can filter and rank by those signals accurately.

### Strategy B — Professor-Level (COMPARISON)

**Chunk size:** All reviews for one professor combined (avg 599 words, min 135, max 1443 words)
**Overlap:** None
**Rationale:** Professor-level chunks are better for broad "who is this professor overall?" questions, because the LLM sees all opinions in one context window and can synthesize a holistic picture. However, this strategy loses per-review granularity — you can no longer filter by course code or individual rating, and long chunks for popular professors (e.g., Sarah Wolff at 1,443 words) may approach MiniLM's 256-token input limit, causing truncation during embedding.

### Your Decision

**I chose Strategy A because:**
The primary use case is students asking targeted questions — about a specific course, a specific teaching trait, or a difficulty comparison between professors. Strategy A gives the retriever more fine-grained candidates to rank by semantic distance, and each chunk carries clean metadata for filtering. Strategy B would be preferable if most queries were "give me everything about Professor X," but that is better handled by filtering Strategy A chunks by professor name.

**Evidence from chunk inspection:**
- Total Strategy A chunks: **131**
- Average word count: **58 words**
- Shortest chunk: **21 words**
- Longest chunk: **91 words**
- Some very short chunks (< 30 words) contain minimal content like single-sentence dismissals. These are valid opinions but may be weak retrieval matches — they surface less context for the LLM than longer reviews.

---

## 4. Embedding Model

**Model:** all-MiniLM-L6-v2 (sentence-transformers)

**Why this model:**
Review chunks are short (avg 58 words), which is exactly the range where MiniLM-L6-v2 performs well — it was trained on sentence pairs and short paragraphs, not long documents. It produces 384-dimensional embeddings quickly enough to embed all 131 chunks in seconds on CPU, which matters for a course project without GPU access. For a corpus this size, semantic accuracy is more important than throughput, and MiniLM handles subjective, opinion-style text (e.g., "he's tough but fair") better than keyword-based methods.

**Production tradeoff:**
For a real product with thousands of Denison reviews and multilingual students, upgrading to `text-embedding-3-large` (OpenAI) or `voyage-large-2` (Voyage AI) would meaningfully improve retrieval accuracy — especially for nuanced queries like "who is good for students who struggle with theory?" The tradeoff is cost (API calls per embed vs. free local model) and latency (network round-trip). If the corpus stayed small (< 10,000 reviews), the accuracy gain would likely justify the cost.

---

## 5. Retrieval Design

**top_k default:** 5

**Why top_k = 5:**
With 131 total chunks across 12 professors, retrieving 5 chunks gives the LLM a representative sample without flooding it with noise. Too few (top_k = 2) risks missing a critical review if the query phrasing doesn't perfectly match the stored text. Too many (top_k = 10) could pull in low-similarity reviews that dilute the answer or introduce contradictory signals the LLM can't reconcile well. Five is a practical middle ground that fits comfortably within the LLM's context window at ~300 words of retrieved text.

**Metadata filtering strategy:**
- **Professor filter:** Use when the student already knows who they're asking about — e.g., "Is Sarah Wolff good for MATH300?" Bypasses semantic search for that field and returns only reviews about that professor, eliminating noise from other professors with similar review language.
- **Department filter:** Use when comparing professors within a field — e.g., "Which Math professor is most accessible?" Returns only reviews from Mathematics, ensuring the answer isn't contaminated by CS or Physics reviews.
- **Min rating filter:** Use when the student wants to exclude low-rated professors from a recommendation — e.g., filtering to ≥ 4.0★ removes May Mei (2.8★), Zhe Wang (3.0★), Robert Viator (3.2★), and Anthony Bonifonte (3.2★) from the candidate pool.

**Distance threshold for "strong match":**
After testing, distance < 0.4 reliably surfaces reviews that directly address the query topic. Queries about specific professors (e.g., "Ashwin Lall workload") return distances of 0.2–0.35 for highly relevant chunks. Some borderline matches at distance 0.45–0.55 were still somewhat relevant — they used related vocabulary (e.g., "homework" vs. "assignments") but weren't the best examples. The 0.4 threshold is a reasonable flag, not a hard cutoff.

---

## 6. Grounded Generation

**Model:** llama-3.3-70b-versatile (Groq)

**How grounding is enforced:**
1. System prompt explicitly forbids use of outside knowledge.
2. Retrieved context is passed in the user message, not the system message.
3. Source attribution is programmatically appended — not left to the LLM.
4. Temperature = 0.1 to reduce creative drift.

**Out-of-scope handling:**
When asked "What are the best restaurants near Denison University?", the system retrieved the top 5 semantically closest reviews (which matched on unrelated keywords), but the LLM correctly responded: *"I don't have enough information in the collected documents to answer that."* The system did not hallucinate restaurant names or use outside knowledge, confirming the grounding prompt works as intended.

---

## 7. Evaluation Plan

**5 test questions I will use:**

1. **What do students think about Sarah Wolff's teaching style?**
   - Expected answer: Reviewers describe her as an amazing lecturer who gives great feedback and is accessible outside class; several mention her courses are tough but rewarding. (Based on 25 reviews with avg 4.5★ and tags: amazing lectures, gives good feedback, accessible outside class.)

2. **How hard is DA101?**
   - Expected answer: DA101 reviews describe difficulty ratings of 3–5/5 depending on the professor. Alexandre Scarcioffolo's section is rated tough (avg difficulty 4.1/5) but manageable with office hours. Grading on few high-stakes items is a recurring pattern.

3. **Which Computer Science professor is best for beginners?**
   - Expected answer: Ashwin Lall consistently rated the best CS professor (avg 4.9★, difficulty 2.8/5). Reviews across CS110, COMP101, and CS111 describe him as fair, clear, and accessible for students new to coding.

4. **Is May Mei a difficult professor?**
   - Expected answer: Reviews reflect significant frustration — avg 2.8★ across 9 reviews. Students mention tough grading and unclear expectations. This tests whether the system surfaces negative reviews accurately without softening them.

5. **What gyms are near Denison University?**
   - Expected answer: The system should respond with the out-of-scope refusal: *"I don't have enough information in the collected documents to answer that."*

**Known anticipated failure:**
A question like "Which professor is best for students who are bad at math?" will likely underperform — reviews don't use that exact framing, and the embedding distance to chunks about "beginner-friendly" or "accessible" professors may be too high to surface the most relevant results. The likely failure mode is a retrieval miss: semantically the right professors exist in the data, but the query vocabulary doesn't align well with how reviewers wrote.

---

## 8. Stretch Features

### Implemented:

**A. Chunking Strategy Comparison**
Both strategies are implemented in `src/chunk.py`. A comparison table is printed when running `python -m src.chunk`. ChromaDB stores both collections (`review_level` and `professor_level`) for live comparison via the Gradio UI.

**B. Metadata Filtering**
`src/retrieve.py` implements `filter_by_professor()`, `filter_by_department()`, and `filter_by_min_rating()`. The Gradio interface exposes all three as UI inputs. The CLI supports them via flags.

---

## 9. Known Limitations

- **Narrow departmental coverage:** Only Mathematics, Computer Science, and Physics are represented. Students in Humanities, Social Sciences, Biology, or other departments get no useful results — the system will either retrieve weakly-matched reviews or correctly refuse, but cannot help those users.
- **Sparse coverage for some professors:** Mason Shero (2 reviews) and Sarah Supp (5 reviews) have too little data to support reliable synthesis. A question specifically about them may return a single review's opinion as if it were consensus.
- **Static snapshot:** All 131 reviews were collected between April 2023 and September 2024. New reviews, professor changes, or course restructuring after that date are not reflected. The system can't tell a student if a professor's course has changed significantly.

---

## 10. What I Would Do With More Time

- **Expand coverage to more departments:** Add 10–15 professors from Biology, Economics, Psychology, and History — the departments with the highest enrollment outside STEM. This would make the system genuinely useful to most Denison students, not just STEM majors.
- **Hybrid search (BM25 + semantic):** Exact professor name matches like "Ashwin Lall" sometimes score lower than expected in pure semantic search because the name itself isn't semantically meaningful. Adding a BM25 keyword layer and combining scores would make named-entity queries more reliable.
- **Semester-aware retrieval:** Tag each review with its academic semester and let users filter by recency (e.g., "reviews from the last two semesters only"). This would prevent old reviews from dominating answers for professors whose courses have changed significantly.
