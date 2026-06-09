"""
src/retrieve.py
===============
PURPOSE: Query ChromaDB for relevant chunks using semantic similarity.
         Supports optional metadata filtering (stretch feature).

RETRIEVAL APPROACH:
  1. Encode the user query with the same model used for embedding.
  2. Query ChromaDB for the top-k most similar chunks by cosine distance.
  3. Optionally apply metadata filters (professor name, department, min rating).
  4. Return a list of RetrievalResult objects with text, metadata, and distance.

DISTANCE SCORES:
  ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
  Similarity = 1 - distance. Scores below 0.4 distance are strong matches.

METADATA FILTERING (stretch feature):
  ChromaDB supports pre-filtering by metadata before vector search.
  This lets users ask "what do students think about Dr. Chen specifically?"
  and get only Dr. Chen's chunks, not similar text about other professors.
  Filters use ChromaDB's `where` clause syntax.

Run with:
    python -m src.retrieve

Expected output:
    Test queries printed with ranked results.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = Path("chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Shared model instance (loaded once, reused) ────────────────────────────────
# We use a module-level variable so the model is only loaded once per process.
_model: Optional[SentenceTransformer] = None
_client: Optional[chromadb.PersistentClient] = None


def get_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer model, loading it if needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_client() -> chromadb.PersistentClient:
    """Return the shared ChromaDB client, connecting if needed."""
    global _client
    if _client is None:
        if not CHROMA_DB_PATH.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DB_PATH}.\n"
                "Run python -m src.embed first."
            )
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _client


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single retrieved chunk with all its context.
    Using a dataclass makes it easy to pass structured results to generate.py.
    """
    chunk_id: str
    text: str
    distance: float          # Cosine distance (lower = more similar)
    similarity: float        # 1 - distance (higher = more similar)
    professor_name: str
    department: str
    course_code: str
    star_rating: float
    difficulty_rating: float
    tags: str
    source: str
    rank: int                 # 1-indexed rank in the result list
    metadata: dict = field(default_factory=dict)  # Full metadata dict

    @property
    def is_strong_match(self) -> bool:
        """True if distance < 0.4 (similarity > 0.6) — heuristic threshold."""
        return self.distance < 0.4


# ── Core retrieval function ────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 5,
    collection_name: str = "review_level",
    where: Optional[dict] = None,
) -> list[RetrievalResult]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query:            The user's question or search string.
        top_k:            Number of chunks to return.
        collection_name:  Which ChromaDB collection to search.
                          "review_level" (default) or "professor_level".
        where:            Optional ChromaDB metadata filter.
                          Example: {"professor_name": {"$eq": "Dr. Sarah Chen"}}
                          Example: {"star_rating": {"$gte": 4.0}}
                          Example: {"department": {"$eq": "Computer Science"}}

    Returns:
        List of RetrievalResult objects, sorted by distance ascending.
    """
    model = get_model()
    client = get_client()

    # Get the collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(
            f"Collection '{collection_name}' not found in ChromaDB.\n"
            "Run python -m src.embed to create it."
        )

    # Encode the query
    query_embedding = model.encode([query])[0].tolist()

    # Build query kwargs
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),  # Don't ask for more than exists
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    # Execute query
    raw = collection.query(**query_kwargs)

    # Parse results
    results = []
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        result = RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            distance=round(distance, 4),
            similarity=round(1 - distance, 4),
            professor_name=metadata.get("professor_name", "Unknown"),
            department=metadata.get("department", "Unknown"),
            course_code=metadata.get("course_code", "Unknown"),
            star_rating=float(metadata.get("star_rating", 0)),
            difficulty_rating=float(metadata.get("difficulty_rating", 0)),
            tags=metadata.get("tags", ""),
            source=metadata.get("source", ""),
            rank=rank,
            metadata=metadata,
        )
        results.append(result)

    return results


# ── Metadata filter helpers (stretch feature) ─────────────────────────────────

def filter_by_professor(professor_name: str) -> dict:
    """
    Build a ChromaDB where-clause to filter by professor name.
    Use exact match (case-sensitive — match how names are stored).

    Example:
        filter_by_professor("Dr. Sarah Chen")
        → {"professor_name": {"$eq": "Dr. Sarah Chen"}}
    """
    return {"professor_name": {"$eq": professor_name}}


def filter_by_min_rating(min_rating: float) -> dict:
    """
    Build a ChromaDB where-clause to filter to reviews above a star threshold.

    Example:
        filter_by_min_rating(4.0)
        → {"star_rating": {"$gte": 4.0}}
    """
    return {"star_rating": {"$gte": min_rating}}


def filter_by_department(department: str) -> dict:
    """
    Build a ChromaDB where-clause to filter by department.

    Example:
        filter_by_department("Computer Science")
        → {"department": {"$eq": "Computer Science"}}
    """
    return {"department": {"$eq": department}}


# ── Display helpers ────────────────────────────────────────────────────────────

def print_results(results: list[RetrievalResult], query: str) -> None:
    """
    Print retrieval results in a readable format for debugging.
    Shows rank, distance, source, and a text preview.
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    if not results:
        print("  No results found.")
        return

    for r in results:
        match_label = "✓ STRONG" if r.is_strong_match else "~ WEAK"
        print(f"\n  Rank {r.rank} | Distance: {r.distance:.4f} | Similarity: {r.similarity:.4f} | {match_label}")
        print(f"  Professor: {r.professor_name} | Course: {r.course_code} ({r.department})")
        print(f"  Rating: {r.star_rating}/5 | Difficulty: {r.difficulty_rating}/5 | Tags: {r.tags}")
        print(f"  Chunk ID: {r.chunk_id}")
        print(f"  Text preview: {r.text[:200]}...")

    print(f"\n  Relevance notes:")
    strong = [r for r in results if r.is_strong_match]
    weak = [r for r in results if not r.is_strong_match]
    print(f"  - {len(strong)} strong matches (distance < 0.4)")
    print(f"  - {len(weak)} weak matches (distance ≥ 0.4)")
    print("=" * 60)


# ── Test queries ───────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "query": "Which CS professors are good at explaining concepts clearly?",
        "where": None,
        "note": "Broad semantic query — should retrieve CS professors with 'clear' tags or clarity mentions.",
    },
    {
        "query": "How hard is Dr. Sarah Chen's courses?",
        "where": filter_by_professor("Dr. Sarah Chen"),
        "note": "Metadata-filtered: only Dr. Chen's reviews. Tests filter + semantic combo.",
    },
    {
        "query": "What professors are good for students who are nervous about math?",
        "where": None,
        "note": "Semantic inference query — 'accessible' and 'caring' tags should surface.",
    },
    {
        "query": "Are there any professors known for tough grading?",
        "where": None,
        "note": "Should match reviews with 'tough grader' tag and related language.",
    },
    {
        "query": "Who are the best economics professors at Denison?",
        "where": filter_by_department("Economics"),
        "note": "Department filter + quality query. Should only return Econ results.",
    },
    {
        "query": "What is the workload like for upper-level biology courses?",
        "where": filter_by_department("Biology"),
        "note": "Department filter on biology. Tests specificity.",
    },
]


if __name__ == "__main__":
    print("[retrieve] Running retrieval test queries...")
    print("[retrieve] Loading model and ChromaDB...")

    for test in TEST_QUERIES:
        results = retrieve(
            query=test["query"],
            top_k=5,
            collection_name="review_level",
            where=test.get("where"),
        )
        print_results(results, test["query"])
        print(f"  Test note: {test['note']}")

    print("\n[retrieve] ✓ Retrieval tests complete. Proceed to: python -m src.generate")
