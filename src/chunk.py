"""
src/chunk.py
============
PURPOSE: Split reviews into chunks suitable for embedding and retrieval.

STRETCH FEATURE: Two chunking strategies are implemented and compared.

WHY CHUNKING MATTERS FOR REVIEWS:
Each review is already 50–200 words — a natural unit of meaning.
The risk is not "too big" but "too small" (single reviews may not have
enough context to answer "what do students think about Prof X overall").

STRATEGY A — Review-Level (one chunk per review):
  - Each individual review becomes one chunk.
  - Preserves the voice and context of a single reviewer.
  - Best for specific questions: "Is Dr. Chen tough on grading?"
  - Weakness: short reviews (< 50 words) may be too thin for the LLM.

STRATEGY B — Professor-Level (combine all reviews per professor):
  - All reviews for one professor are combined into one chunk.
  - Better for aggregate questions: "What is Dr. Walsh like overall?"
  - Weakness: loses individual review metadata (date, course, rating).
  - Chunks may be 600–1200 words — at the upper limit for MiniLM.

CHOSEN DEFAULT: Strategy A (review-level) with a combine step for very
short reviews. This preserves metadata per chunk and handles the
specific-question use case better. Strategy B is provided for comparison.

Run with:
    python -m src.chunk

Expected output:
    Strategy A: 50 chunks saved → data/chunks/chunks_review_level.jsonl
    Strategy B: 10 chunks saved → data/chunks/chunks_professor_level.jsonl
    Chunk inspection printed.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
CLEANED_CSV = Path("data/cleaned/denison_reviews_cleaned.csv")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_REVIEW_LEVEL = CHUNKS_DIR / "chunks_review_level.jsonl"
CHUNKS_PROFESSOR_LEVEL = CHUNKS_DIR / "chunks_professor_level.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_source_id(professor_name: str) -> str:
    """
    Convert "Dr. Sarah Chen" → "dr_sarah_chen" for use in chunk IDs and filenames.
    """
    return re.sub(r"[^a-z0-9]+", "_", professor_name.lower()).strip("_")


def build_chunk_id(source_id: str, strategy: str, index: int) -> str:
    """
    Build a unique chunk ID like: dr_sarah_chen_review_000
    The strategy tag makes it easy to tell chunks apart in ChromaDB.
    """
    return f"{source_id}_{strategy}_{index:03d}"


# ── Strategy A: Review-Level Chunking ─────────────────────────────────────────

def chunk_by_review(df: pd.DataFrame) -> list[dict]:
    """
    One chunk per individual review row.

    Each chunk includes:
    - The review text (with a header line for readability)
    - All metadata fields: professor, department, course, rating, tags, date

    The header line ("Professor: Dr. Chen | Course: CS 181 | Rating: 4.5/5")
    helps the LLM and retriever understand context without needing to look
    elsewhere. It's embedded AS PART OF the chunk text.

    Returns a list of chunk dicts in the JSONL format.
    """
    chunks = []

    for idx, row in df.iterrows():
        source_id = make_source_id(row["professor_name"])
        chunk_id = build_chunk_id(source_id, "review", len(chunks))

        # Build a readable header that gets embedded with the text
        # This improves retrieval: "who is tough?" will match chunks
        # that mention difficulty even without a separate metadata filter.
        tags_val = str(row['tags']) if str(row['tags']).lower() != 'nan' else ''
        header = (
            f"Professor: {row['professor_name']} | "
            f"Course: {row['course_code']} ({row['department']}) | "
            f"Rating: {row['star_rating']}/5 | "
            f"Difficulty: {row['difficulty_rating']}/5 | "
            f"Tags: {tags_val}"
        )

        # Full text = header + blank line + review body
        full_text = header + "\n\n" + row["review_text"]

        chunk = {
            "chunk_id": chunk_id,
            "source": f"{source_id}.txt",          # logical source name
            "source_path": "data/raw/denison_reviews.csv",
            "strategy": "review_level",
            "chunk_index": len(chunks),
            "professor_name": row["professor_name"],
            "department": row["department"],
            "course_code": row["course_code"],
            "star_rating": float(row["star_rating"]),
            "difficulty_rating": float(row["difficulty_rating"]),
            "review_date": str(row["review_date"]),
            "tags": str(row['tags']) if str(row['tags']).lower() != 'nan' else '',
            "text": full_text,
            "word_count": len(full_text.split()),
        }
        chunks.append(chunk)

    return chunks


# ── Strategy B: Professor-Level Chunking ──────────────────────────────────────

def chunk_by_professor(df: pd.DataFrame) -> list[dict]:
    """
    One chunk per professor: combine all reviews into a single document.

    Format:
        --- Review 1 (CS 181, 4.5/5, tough grader, helpful) ---
        <review text>

        --- Review 2 (CS 271, 5.0/5, amazing lectures) ---
        <review text>

    This approach answers aggregate questions better ("what is Dr. Chen like
    overall?") but loses per-review metadata. The avg_star_rating is computed
    across all reviews for the professor.

    WARNING: For professors with many long reviews, chunks may exceed 1000
    words, which stretches all-MiniLM-L6-v2's ideal input length (~256 tokens).
    """
    chunks = []
    grouped = df.groupby("professor_name")

    for prof_name, group in grouped:
        source_id = make_source_id(prof_name)
        chunk_id = build_chunk_id(source_id, "professor", len(chunks))

        # Build combined text with per-review mini-headers
        parts = []
        for i, (_, row) in enumerate(group.iterrows()):
            mini_header = (
                f"--- Review {i+1} "
                f"({row['course_code']}, "
                f"{row['star_rating']}/5 stars, "
                f"difficulty {row['difficulty_rating']}/5, "
                f"tags: {row['tags']}) ---"
            )
            parts.append(mini_header + "\n" + row["review_text"])

        # Add a professor summary header at the top
        avg_stars = group["star_rating"].mean()
        avg_diff = group["difficulty_rating"].mean()
        summary_header = (
            f"Professor: {prof_name} | "
            f"Department: {group['department'].iloc[0]} | "
            f"Avg Rating: {avg_stars:.1f}/5 | "
            f"Avg Difficulty: {avg_diff:.1f}/5 | "
            f"Total Reviews: {len(group)}"
        )

        full_text = summary_header + "\n\n" + "\n\n".join(parts)

        # For metadata: store all unique courses and tags
        # str(tags_str) handles NaN values (which become the string "nan")
        all_tags = list(set(
            tag for tags_str in group["tags"]
            for tag in str(tags_str).split(",")
            if tag.strip() and tag.strip().lower() != "nan"
        ))
        all_courses = list(group["course_code"].unique())

        chunk = {
            "chunk_id": chunk_id,
            "source": f"{source_id}.txt",
            "source_path": "data/raw/denison_reviews.csv",
            "strategy": "professor_level",
            "chunk_index": len(chunks),
            "professor_name": prof_name,
            "department": group["department"].iloc[0],
            "course_codes": ",".join(all_courses),
            "avg_star_rating": round(avg_stars, 2),
            "avg_difficulty_rating": round(avg_diff, 2),
            "review_count": len(group),
            "all_tags": ",".join(all_tags),
            "text": full_text,
            "word_count": len(full_text.split()),
        }
        chunks.append(chunk)

    return chunks


# ── Save and Inspect ───────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], out_path: Path) -> None:
    """Save chunks to a JSONL file (one JSON object per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[chunk] Saved {len(chunks)} chunks → {out_path}")


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inspect_chunks(chunks: list[dict], strategy_name: str, n_sample: int = 5) -> None:
    """
    Print a detailed inspection of the chunk set.
    Run this after chunking to verify quality before embedding.

    Checks:
    - Total count
    - Average word count
    - Shortest and longest chunks (look for fragments or oversized blocks)
    - 5 random samples
    """
    word_counts = [c["word_count"] for c in chunks]
    avg_wc = sum(word_counts) / len(word_counts)
    min_wc = min(word_counts)
    max_wc = max(word_counts)
    min_chunk = min(chunks, key=lambda c: c["word_count"])
    max_chunk = max(chunks, key=lambda c: c["word_count"])

    print(f"\n{'='*60}")
    print(f"CHUNK INSPECTION — {strategy_name.upper()}")
    print(f"{'='*60}")
    print(f"Total chunks:      {len(chunks)}")
    print(f"Avg word count:    {avg_wc:.1f}")
    print(f"Min word count:    {min_wc}  (chunk: {min_chunk['chunk_id']})")
    print(f"Max word count:    {max_wc}  (chunk: {max_chunk['chunk_id']})")

    print(f"\n--- SHORTEST CHUNK ({min_wc} words) ---")
    print(min_chunk["text"][:300])

    print(f"\n--- LONGEST CHUNK ({max_wc} words) ---")
    print(max_chunk["text"][:400] + "..." if max_chunk["word_count"] > 60 else max_chunk["text"])

    print(f"\n--- {n_sample} RANDOM SAMPLE CHUNKS ---")
    samples = random.sample(chunks, min(n_sample, len(chunks)))
    for i, chunk in enumerate(samples):
        print(f"\n[Sample {i+1}] chunk_id: {chunk['chunk_id']} | {chunk['word_count']} words")
        print(f"  Source: {chunk['source']}")
        print(f"  Professor: {chunk['professor_name']}")
        print(f"  Text preview: {chunk['text'][:200]}...")
    print("=" * 60)


def compare_strategies(chunks_a: list[dict], chunks_b: list[dict]) -> None:
    """
    Print a side-by-side comparison of the two chunking strategies.
    This is the stretch feature deliverable.
    """
    wc_a = [c["word_count"] for c in chunks_a]
    wc_b = [c["word_count"] for c in chunks_b]

    print(f"\n{'='*60}")
    print("CHUNKING STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Strategy A (Review)':<20} {'Strategy B (Professor)'}")
    print(f"{'-'*35} {'-'*20} {'-'*22}")
    print(f"{'Total chunks':<35} {len(chunks_a):<20} {len(chunks_b)}")
    print(f"{'Avg words/chunk':<35} {sum(wc_a)/len(wc_a):<20.1f} {sum(wc_b)/len(wc_b):.1f}")
    print(f"{'Min words':<35} {min(wc_a):<20} {min(wc_b)}")
    print(f"{'Max words':<35} {max(wc_a):<20} {max(wc_b)}")
    print(f"{'Per-review metadata?':<35} {'Yes':<20} {'No (aggregated)'}")
    print(f"{'Best for':<35} {'Specific queries':<20} {'Aggregate queries'}")
    print(f"{'MiniLM token risk?':<35} {'Low':<20} {'Medium-High'}")
    print(f"\nRECOMMENDATION: Strategy A (review-level) is the default")
    print(f"because it preserves per-review metadata (star rating, course,")
    print(f"tags) that enables metadata filtering, and it keeps chunks")
    print(f"within all-MiniLM-L6-v2's effective input range (~256 tokens).")
    print(f"Strategy B is available in ChromaDB as collection 'professor_level'")
    print(f"for comparison during evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    # Load cleaned data
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(
            "Run python -m src.clean first to generate data/cleaned/denison_reviews_cleaned.csv"
        )

    df = pd.read_csv(CLEANED_CSV)
    print(f"[chunk] Loaded {len(df)} cleaned reviews.")

    # Strategy A: Review-level
    chunks_a = chunk_by_review(df)
    save_chunks(chunks_a, CHUNKS_REVIEW_LEVEL)
    inspect_chunks(chunks_a, "Strategy A: Review-Level")

    # Strategy B: Professor-level
    chunks_b = chunk_by_professor(df)
    save_chunks(chunks_b, CHUNKS_PROFESSOR_LEVEL)
    inspect_chunks(chunks_b, "Strategy B: Professor-Level")

    # Comparison
    compare_strategies(chunks_a, chunks_b)

    print("\n[chunk] ✓ Chunking complete. Proceed to: python -m src.embed")