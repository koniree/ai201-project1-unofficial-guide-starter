"""
src/embed.py
============
PURPOSE: Encode all chunks with sentence-transformers and store them
         in ChromaDB with full metadata for retrieval and filtering.

EMBEDDING MODEL: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Fast, lightweight, good semantic understanding for English text
  - Effective input: ~256 tokens (~180 words). Our review-level chunks
    average ~120 words including the header, so we are well within range.
  - Production tradeoff: For a production system, you might use
    text-embedding-3-small (OpenAI) for better multilingual support
    or all-mpnet-base-v2 for higher accuracy at the cost of speed.
    all-MiniLM-L6-v2 is the right choice for a local, fast prototype.

ChromaDB COLLECTIONS:
  - "review_level"     → Strategy A chunks (default, used for querying)
  - "professor_level"  → Strategy B chunks (for comparison)

Run with:
    python -m src.embed

Expected output:
    Loading model all-MiniLM-L6-v2...
    Embedding 50 chunks (review_level)...
    Stored 50 chunks in ChromaDB collection 'review_level'.
    Embedding 10 chunks (professor_level)...
    Stored 10 chunks in ChromaDB collection 'professor_level'.
    ✓ Embedding complete.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
CHUNKS_DIR = Path("data/chunks")
CHUNKS_REVIEW_LEVEL = CHUNKS_DIR / "chunks_review_level.jsonl"
CHUNKS_PROFESSOR_LEVEL = CHUNKS_DIR / "chunks_professor_level.jsonl"
CHROMA_DB_PATH = Path("chroma_db")

# ── Model ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── ChromaDB metadata keys allowed per strategy ───────────────────────────────
# ChromaDB requires all metadata values to be str, int, float, or bool.
# We map chunk dict keys → metadata dict keys here.
#
# IMPORTANT: "text" is stored as the document body, not in metadata.
# Only use metadata for fields you want to filter on.

REVIEW_LEVEL_METADATA_KEYS = [
    "professor_name",
    "department",
    "course_code",
    "star_rating",
    "difficulty_rating",
    "review_date",
    "tags",
    "strategy",
    "chunk_index",
    "source",
    "word_count",
]

PROFESSOR_LEVEL_METADATA_KEYS = [
    "professor_name",
    "department",
    "course_codes",
    "avg_star_rating",
    "avg_difficulty_rating",
    "review_count",
    "all_tags",
    "strategy",
    "chunk_index",
    "source",
    "word_count",
]


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}\n"
            "Run python -m src.chunk first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_metadata(chunk: dict, allowed_keys: list[str]) -> dict:
    """
    Extract only the allowed metadata keys from a chunk dict.
    Ensures all values are ChromaDB-safe types (str, int, float, bool).
    Missing keys default to empty string.
    """
    metadata = {}
    for key in allowed_keys:
        val = chunk.get(key, "")
        # ChromaDB does not accept None
        if val is None:
            val = ""
        # Cast to safe types
        if isinstance(val, (int, float, bool)):
            metadata[key] = val
        else:
            metadata[key] = str(val)
    return metadata


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.
    If reset=True, delete existing collection first (use for re-embedding).

    We do NOT pass an embedding_function to ChromaDB because we compute
    embeddings ourselves with SentenceTransformer. This gives us full
    control over the embedding process and lets us batch efficiently.
    """
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[embed] Deleted existing collection '{collection_name}'.")
        except Exception:
            pass  # Collection didn't exist, that's fine

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine distance for similarity
    )
    return collection


def embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
    metadata_keys: list[str],
    batch_size: int = 32,
) -> None:
    """
    Embed all chunks and store them in ChromaDB.

    We process in batches to avoid memory issues with large datasets.
    Each chunk is stored as:
      - id:        chunk_id string
      - embedding: 384-dim vector from all-MiniLM-L6-v2
      - document:  chunk text (what ChromaDB returns on retrieval)
      - metadata:  dict of filterable fields

    Args:
        chunks:        list of chunk dicts from JSONL
        collection:    ChromaDB collection to store into
        model:         SentenceTransformer model
        metadata_keys: which fields to store as metadata
        batch_size:    how many chunks to embed at once
    """
    total = len(chunks)
    print(f"[embed] Embedding {total} chunks into '{collection.name}'...")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        # Compute embeddings (returns numpy array of shape [batch_size, 384])
        embeddings = model.encode(texts, show_progress_bar=False)

        # Prepare ChromaDB inputs
        ids = [chunk["chunk_id"] for chunk in batch]
        metadatas = [extract_metadata(chunk, metadata_keys) for chunk in batch]
        embeddings_list = embeddings.tolist()  # ChromaDB expects list of lists

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
        )

        end = min(batch_start + batch_size, total)
        print(f"[embed]   Stored batch {batch_start+1}–{end} / {total}")

    print(f"[embed] ✓ Stored {total} chunks in collection '{collection.name}'.")
    print(f"[embed]   Collection now has {collection.count()} total documents.")


def verify_collection(collection: chromadb.Collection, n: int = 2) -> None:
    """
    Quick verification: peek at n documents from the collection.
    Confirm IDs, metadata, and text look right before moving to retrieval.
    """
    print(f"\n[embed] Verification peek ({n} items from '{collection.name}'):")
    results = collection.peek(limit=n)
    for i in range(min(n, len(results["ids"]))):
        print(f"  ID:       {results['ids'][i]}")
        print(f"  Metadata: {results['metadatas'][i]}")
        print(f"  Text[:100]: {results['documents'][i][:100]}...")
        print()


if __name__ == "__main__":
    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"[embed] Loading model '{EMBEDDING_MODEL_NAME}'...")
    print("[embed] (First run will download ~90MB — subsequent runs are instant.)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"[embed] Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    print(f"[embed] ChromaDB client connected at {CHROMA_DB_PATH}")

    # ── Strategy A: Review-Level ───────────────────────────────────────────────
    chunks_a = load_chunks(CHUNKS_REVIEW_LEVEL)
    collection_a = get_or_create_collection(client, "review_level", reset=True)
    embed_and_store(chunks_a, collection_a, model, REVIEW_LEVEL_METADATA_KEYS)
    verify_collection(collection_a)

    # ── Strategy B: Professor-Level ────────────────────────────────────────────
    chunks_b = load_chunks(CHUNKS_PROFESSOR_LEVEL)
    collection_b = get_or_create_collection(client, "professor_level", reset=True)
    embed_and_store(chunks_b, collection_b, model, PROFESSOR_LEVEL_METADATA_KEYS)
    verify_collection(collection_b)

    print("\n[embed] ✓ Embedding complete. Proceed to: python -m src.retrieve")
