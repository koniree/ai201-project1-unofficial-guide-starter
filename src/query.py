"""
src/query.py
============
PURPOSE: Command-line interface for asking a single question end-to-end.

Run with:
    python -m src.query "Which professor is best for intro CS?"
    python -m src.query "How hard is Dr. Sharma?" --top_k 3
    python -m src.query "Best econ prof?" --professor "Dr. Amara Osei"
    python -m src.query "Good professors?" --min_rating 4.5
    python -m src.query "Tell me about chemistry" --department "Chemistry"
    python -m src.query "Which strategy is better?" --collection professor_level

Expected output:
    QUESTION: Which professor is best for intro CS?
    [Retrieving top 5 chunks from collection 'review_level'...]
    [Retrieved 5 chunks in 0.12s]

    ANSWER:
    Based on student reviews...

    --- Retrieved Sources: ---
      • Dr. Sarah Chen — CS 181 ...
"""

import argparse
import time

from src.generate import ask
from src.retrieve import (
    filter_by_department,
    filter_by_min_rating,
    filter_by_professor,
    print_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the Unofficial Guide RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.query "Which CS professor explains things most clearly?"
  python -m src.query "How hard is Dr. Sharma?" --professor "Dr. Priya Sharma"
  python -m src.query "Best intro courses?" --min_rating 4.0
  python -m src.query "Biology workload?" --department "Biology" --top_k 3
  python -m src.query "About Dr. Chen" --show_chunks
        """,
    )
    parser.add_argument("question", type=str, help="Your question about Denison professors.")
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--collection", type=str, default="review_level",
        choices=["review_level", "professor_level"],
        help="Which ChromaDB collection to use (default: review_level).",
    )
    parser.add_argument(
        "--professor", type=str, default=None,
        help='Filter results to a specific professor. Example: "Dr. Sarah Chen"',
    )
    parser.add_argument(
        "--department", type=str, default=None,
        help='Filter results to a specific department. Example: "Computer Science"',
    )
    parser.add_argument(
        "--min_rating", type=float, default=None,
        help="Filter to reviews with star_rating >= this value. Example: 4.0",
    )
    parser.add_argument(
        "--show_chunks", action="store_true",
        help="Print retrieved chunks in detail before showing the answer.",
    )
    return parser.parse_args()


def build_where_clause(args) -> dict | None:
    """
    Build a ChromaDB where clause from CLI filter arguments.
    Only one filter can be active at a time in this simple implementation.
    (ChromaDB supports $and/$or for combining — add that if needed.)
    """
    if args.professor:
        return filter_by_professor(args.professor)
    if args.department:
        return filter_by_department(args.department)
    if args.min_rating is not None:
        return filter_by_min_rating(args.min_rating)
    return None


def main():
    args = parse_args()
    where = build_where_clause(args)

    print(f"\nQUESTION: {args.question}")
    if where:
        print(f"FILTER:   {where}")
    print(f"[Retrieving top {args.top_k} chunks from collection '{args.collection}'...]")

    start = time.time()
    answer, results = ask(
        question=args.question,
        top_k=args.top_k,
        collection_name=args.collection,
        where=where,
    )
    elapsed = time.time() - start

    print(f"[Retrieved {len(results)} chunks in {elapsed:.2f}s]\n")

    # Optionally show retrieved chunks in detail
    if args.show_chunks:
        print_results(results, args.question)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print()


if __name__ == "__main__":
    main()
