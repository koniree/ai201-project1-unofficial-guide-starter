"""
src/clean.py
============
PURPOSE: Clean and normalize raw review text before chunking.

Professor reviews from RateMyProfessor can contain:
- HTML entities (&amp; &nbsp; etc.)
- Excess whitespace and newlines
- Unicode artifacts (curly quotes, em-dashes)
- All-caps text (shouting)
- Leading/trailing noise

This module produces data/cleaned/denison_reviews_cleaned.csv.

Run with:
    python -m src.clean

Expected output:
    Cleaned 50 reviews → data/cleaned/denison_reviews_cleaned.csv
"""

import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

from src.ingest import load_reviews

# ── Paths ──────────────────────────────────────────────────────────────────────
CLEANED_DIR = Path("data/cleaned")
CLEANED_CSV = CLEANED_DIR / "denison_reviews_cleaned.csv"


def strip_html(text: str) -> str:
    """
    Remove any HTML tags and decode HTML entities.
    BeautifulSoup handles both: <br> tags and &amp; entities.

    Example:
        "<p>Great professor!&nbsp;</p>" → "Great professor!"
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces, tabs, and newlines into a single space.
    Strips leading/trailing whitespace.

    Example:
        "Great  professor.\n\n  Very helpful." → "Great professor. Very helpful."
    """
    # Replace tabs and newlines with spaces
    text = re.sub(r"[\t\n\r]+", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Replace curly quotes, em-dashes, and other Unicode artifacts
    with their plain ASCII equivalents.

    This prevents embedding models from seeing "weird" token splits.
    """
    replacements = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def clean_review_text(text: str) -> str:
    """
    Apply all cleaning steps in order to a single review string.

    Steps:
    1. Strip HTML
    2. Normalize Unicode
    3. Normalize whitespace

    We intentionally do NOT:
    - Remove stopwords (embedding models handle these)
    - Lowercase everything (proper nouns like professor names matter)
    - Remove punctuation (sentence boundaries help the LLM)
    """
    text = strip_html(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def clean_tags(tags: str) -> str:
    """
    Normalize the comma-separated tags string.
    Lowercases, strips spaces around commas, deduplicates.

    Example:
        "Tough Grader , helpful,  Helpful" → "helpful,tough grader"
    """
    if not tags or tags.strip() == "":
        return ""
    parts = [t.strip().lower() for t in tags.split(",")]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)
    return ",".join(deduped)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning to all relevant columns in the DataFrame.
    Returns a new DataFrame with cleaned columns.
    """
    df = df.copy()

    # Clean the review text (main content)
    df["review_text"] = df["review_text"].apply(clean_review_text)

    # Clean tags
    df["tags"] = df["tags"].apply(clean_tags)

    # Drop any reviews that became empty after cleaning
    before = len(df)
    df = df[df["review_text"].str.len() > 20]  # Must be at least 20 chars
    dropped = before - len(df)
    if dropped > 0:
        print(f"[clean] Dropped {dropped} reviews that were too short after cleaning.")

    return df


def save_cleaned(df: pd.DataFrame, out_path: Path = CLEANED_CSV) -> None:
    """Save cleaned DataFrame to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[clean] Saved {len(df)} cleaned reviews → {out_path}")


def show_cleaning_diff(raw_df: pd.DataFrame, clean_df: pd.DataFrame, n: int = 3) -> None:
    """
    Show before/after comparison for n random reviews.
    Useful for verifying the cleaning is working correctly.
    """
    print("\n" + "=" * 60)
    print(f"CLEANING DIFF (showing {n} examples)")
    print("=" * 60)
    sample_idx = raw_df.sample(min(n, len(raw_df)), random_state=42).index
    for i, idx in enumerate(sample_idx):
        raw_text = raw_df.loc[idx, "review_text"]
        clean_text = clean_df.loc[idx, "review_text"]
        print(f"\nExample {i+1} — {raw_df.loc[idx, 'professor_name']} / {raw_df.loc[idx, 'course_code']}")
        print(f"  RAW:     {repr(raw_text[:120])}")
        print(f"  CLEANED: {repr(clean_text[:120])}")
    print("=" * 60)


if __name__ == "__main__":
    # Load raw data
    raw_df = load_reviews()

    # Clean it
    clean_df = clean_dataframe(raw_df)

    # Show diff for verification
    show_cleaning_diff(raw_df, clean_df)

    # Save
    save_cleaned(clean_df)

    print("\n[clean] ✓ Cleaning complete. Proceed to: python -m src.chunk")
