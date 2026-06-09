"""
src/ingest.py
=============
PURPOSE: Load the raw CSV of Denison professor reviews, validate its structure,
         and report a summary of what was loaded.

This is the FIRST stage of the pipeline. It does not clean or chunk anything —
it just confirms the data is readable and well-formed.

Run with:
    python -m src.ingest

Expected output:
    Loaded 50 reviews across 10 professors.
    Columns: professor_name, department, course_code, star_rating, ...
    Sample row: ...
    No missing review texts found.
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_CSV = Path("data/raw/denison_reviews.csv")

# ── Required columns ──────────────────────────────────────────────────────────
# These must exist in the CSV or we raise an error immediately.
REQUIRED_COLUMNS = [
    "professor_name",
    "department",
    "course_code",
    "star_rating",
    "difficulty_rating",
    "review_date",
    "tags",
    "review_text",
]


def load_reviews(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Load the reviews CSV and return a cleaned DataFrame.

    Steps:
    1. Read CSV with pandas
    2. Validate all required columns exist
    3. Drop rows with missing review_text (those are unusable)
    4. Normalize string columns (strip whitespace, lowercase professor names)
    5. Cast numeric columns to float
    6. Return the validated DataFrame
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}.\n"
            "Create data/raw/denison_reviews.csv with columns:\n"
            f"  {', '.join(REQUIRED_COLUMNS)}"
        )

    # Step 1: Read
    df = pd.read_csv(csv_path)
    print(f"[ingest] Read {len(df)} rows from {csv_path}")

    # Step 2: Validate columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Step 3: Drop rows with empty review_text
    before = len(df)
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]
    dropped = before - len(df)
    if dropped > 0:
        print(f"[ingest] WARNING: Dropped {dropped} rows with empty review_text.")

    # Step 4: Normalize strings
    # Strip leading/trailing whitespace from all string columns
    for col in ["professor_name", "department", "course_code", "tags", "review_text"]:
        df[col] = df[col].astype(str).str.strip()

    # Normalize professor names to Title Case (catches "dr. chen" vs "Dr. Chen")
    df["professor_name"] = df["professor_name"].str.title()

    # Step 5: Cast numeric columns
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce").fillna(0.0)
    df["difficulty_rating"] = pd.to_numeric(
        df["difficulty_rating"], errors="coerce"
    ).fillna(0.0)

    print(f"[ingest] Validated {len(df)} reviews.")
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the loaded dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total reviews:       {len(df)}")
    print(f"Unique professors:   {df['professor_name'].nunique()}")
    print(f"Unique departments:  {df['department'].nunique()}")
    print(f"Unique courses:      {df['course_code'].nunique()}")
    print(f"Avg star rating:     {df['star_rating'].mean():.2f}")
    print(f"Avg difficulty:      {df['difficulty_rating'].mean():.2f}")

    print("\nReviews per professor:")
    counts = df.groupby("professor_name").size().sort_values(ascending=False)
    for prof, count in counts.items():
        print(f"  {prof:<30} {count} reviews")

    print("\nStar rating distribution:")
    bins = pd.cut(df["star_rating"], bins=[0, 2, 3, 4, 5], labels=["1-2", "2-3", "3-4", "4-5"])
    print(bins.value_counts().sort_index().to_string())

    print("\nSample row (first review):")
    sample = df.iloc[0]
    print(f"  Professor:  {sample['professor_name']}")
    print(f"  Course:     {sample['course_code']} ({sample['department']})")
    print(f"  Rating:     {sample['star_rating']} stars, {sample['difficulty_rating']} difficulty")
    print(f"  Tags:       {sample['tags']}")
    print(f"  Text:       {sample['review_text'][:120]}...")
    print("=" * 60)


if __name__ == "__main__":
    df = load_reviews()
    summarize(df)
    print("\n[ingest] ✓ Ingestion complete. Proceed to: python -m src.clean")
