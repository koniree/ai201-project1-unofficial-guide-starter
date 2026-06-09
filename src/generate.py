"""
src/generate.py
===============
PURPOSE: Generate grounded answers using retrieved chunks as context.

GROUNDING APPROACH:
  The system prompt strictly instructs the LLM to:
  1. Answer ONLY from the retrieved context.
  2. Never use outside knowledge.
  3. Say "I don't have enough information..." if context is insufficient.
  4. Not guess or hallucinate.

  Additionally, source attribution is PROGRAMMATICALLY APPENDED after
  the LLM response — it does not rely on the LLM to format citations.
  This ensures citations are always present and accurate.

MODEL: llama-3.3-70b-versatile via Groq API
API KEY: loaded from .env (GROQ_API_KEY)

Run with:
    python -m src.generate

Expected output:
    Test question answered with retrieved context, then sources appended.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.retrieve import RetrievalResult, retrieve

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Create a .env file with:\n  GROQ_API_KEY=your_key_here\n"
        "Get your key at https://console.groq.com/"
    )

# ── Groq client ───────────────────────────────────────────────────────────────
_groq_client: Groq = None


def get_groq_client() -> Groq:
    """Return the shared Groq client, creating it if needed."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── System prompt ──────────────────────────────────────────────────────────────
# This is the most important single string in the whole pipeline.
# It enforces grounding and prevents hallucination.
SYSTEM_PROMPT = """You are answering questions for The Unofficial Guide — a RAG system \
built from real student reviews of Denison University professors.

STRICT RULES:
1. Answer ONLY from the retrieved context provided. Do not use any outside knowledge.
2. If the context does not contain enough information to answer the question, \
say exactly: "I don't have enough information in the collected documents to answer that."
3. Do not guess, infer beyond the text, or make up details.
4. When answering, reference specific things reviewers said — quote or paraphrase directly.
5. If reviews conflict with each other, acknowledge the disagreement rather than picking one side.
6. Keep your answer concise and useful for a student deciding which professor to take.
7. Do NOT list sources — the system will append them automatically."""


def format_context(results: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.

    Each chunk is labeled with its source professor, course, and rating
    so the LLM can reference them accurately. The label is part of the
    context the LLM receives — it helps the model say "reviewers of
    Dr. Chen for CS 181 noted..." rather than vague references.
    """
    if not results:
        return "No relevant documents were retrieved."

    context_parts = []
    for r in results:
        label = (
            f"[Source {r.rank}: {r.professor_name} | {r.course_code} | "
            f"{r.star_rating}/5 stars | Tags: {r.tags}]"
        )
        context_parts.append(f"{label}\n{r.text}")

    return "\n\n---\n\n".join(context_parts)


def format_sources(results: list[RetrievalResult]) -> str:
    """
    Build the programmatic source attribution block.

    This is ALWAYS appended to the final answer regardless of what the LLM
    generates. It is built from retrieval metadata, not from LLM output.
    This guarantees accurate citations even if the LLM forgets to cite.
    """
    if not results:
        return ""

    lines = ["\n\n---\n📚 **Retrieved Sources:**"]
    seen = set()

    for r in results:
        # Deduplicate: same professor + course shouldn't appear twice
        key = f"{r.professor_name}|{r.course_code}"
        if key in seen:
            continue
        seen.add(key)

        lines.append(
            f"  • {r.professor_name} — {r.course_code} ({r.department}) | "
            f"⭐ {r.star_rating}/5 | 🔑 {r.tags} | "
            f"Similarity: {r.similarity:.2f}"
        )

    return "\n".join(lines)


def generate_answer(
    question: str,
    results: list[RetrievalResult],
    max_tokens: int = 600,
) -> str:
    """
    Generate a grounded answer from retrieved chunks.

    Steps:
    1. Format retrieved chunks as context.
    2. Build the user message: question + context.
    3. Call Groq API with the grounding system prompt.
    4. Programmatically append source attribution.
    5. Return the full response string.

    Args:
        question:   The user's question.
        results:    Retrieved chunks from retrieve().
        max_tokens: Max tokens for LLM response (keep ~600 for concise answers).

    Returns:
        Full response string: LLM answer + source attribution block.
    """
    client = get_groq_client()

    # Step 1: Format context
    context = format_context(results)

    # Step 2: Build user message
    # We include the context IN the user message (not the system message)
    # so the model has a clear separation between instructions and content.
    user_message = f"""Question: {question}

Retrieved Context:
{context}

Based only on the retrieved context above, answer the question."""

    # Step 3: Call Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.1,  # Low temperature = more faithful, less creative
    )

    llm_answer = response.choices[0].message.content.strip()

    # Step 4: Programmatically append sources
    sources_block = format_sources(results)

    # Step 5: Return combined response
    return llm_answer + sources_block


def ask(
    question: str,
    top_k: int = 5,
    collection_name: str = "review_level",
    where: dict = None,
) -> tuple[str, list[RetrievalResult]]:
    """
    End-to-end function: retrieve + generate.
    Returns (answer_string, results_list).

    This is the main entry point used by query.py and app.py.
    """
    results = retrieve(question, top_k=top_k, collection_name=collection_name, where=where)
    answer = generate_answer(question, results)
    return answer, results


# ── Test questions ─────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "What do students think about Dr. Sarah Chen's teaching style?",
    "Which professors are easiest for students who struggle with math?",
    "What is the best way to succeed in Dr. Priya Sharma's chemistry class?",
    "Is Dr. Robert Kim a good professor for intro physics students?",
    "Who teaches the most inspiring courses at Denison?",
    # Out-of-scope test — should trigger the refusal response
    "What are the best restaurants near Denison University?",
]


if __name__ == "__main__":
    print("[generate] Running generation tests...\n")

    for question in TEST_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print("=" * 60)

        answer, results = ask(question, top_k=5)

        print(f"ANSWER:\n{answer}")
        print(f"\n[Retrieved {len(results)} chunks]")

    print("\n[generate] ✓ Generation tests complete. Proceed to: python -m src.query")
