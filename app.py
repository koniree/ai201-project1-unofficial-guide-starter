"""
app.py
======
PURPOSE: Gradio web interface for The Unofficial Guide.

Features:
  - Question input textbox
  - top_k slider (1–10, default 5)
  - Optional professor / department / min-rating filters
  - Answer output panel
  - Sources summary panel
  - Retrieved chunks accordion (debug view)
  - Collection selector (review_level vs professor_level — stretch feature)

Run with:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""

import gradio as gr

from src.generate import generate_answer
from src.retrieve import (
    filter_by_department,
    filter_by_min_rating,
    filter_by_professor,
    retrieve,
)

# ── Domain description shown in the interface ─────────────────────────────────
DOMAIN_DESCRIPTION = """
**The Unofficial Guide** is a searchable knowledge base built from student reviews of Denison University professors.
Ask questions about teaching style, workload, grading, specific courses, or which professors are best for your needs.
All answers are grounded in real student reviews — nothing is made up.
"""


def build_where_clause(professor: str, department: str, min_rating: float) -> dict | None:
    """
    Build a ChromaDB metadata filter from the UI inputs.
    Professor filter takes priority, then department, then rating.
    Returns None if no filters are set.
    """
    professor = professor.strip() if professor else ""
    department = department.strip() if department else ""

    if professor:
        return filter_by_professor(professor)
    if department:
        return filter_by_department(department)
    if min_rating and min_rating > 1.0:
        return filter_by_min_rating(min_rating)
    return None


def format_chunks_for_display(results) -> str:
    """
    Format retrieved chunks for the debug accordion panel.
    Shows chunk ID, distance, and full text.
    """
    if not results:
        return "No chunks retrieved."

    lines = []
    for r in results:
        lines.append(
            f"🔢 Rank {r.rank} | 📏 Distance: {r.distance:.4f} | "
            f"👤 {r.professor_name} | 📚 {r.course_code}\n"
            f"🆔 {r.chunk_id}\n"
            f"{r.text}\n"
            + "─" * 60
        )
    return "\n\n".join(lines)


def format_sources_for_display(results) -> str:
    """
    Format sources for the sources panel (separate from the answer).
    """
    if not results:
        return "No sources retrieved."

    lines = []
    seen = set()
    for r in results:
        key = f"{r.professor_name}|{r.course_code}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"• **{r.professor_name}** — {r.course_code} ({r.department})\n"
            f"  ⭐ {r.star_rating}/5 stars | 🎯 Difficulty: {r.difficulty_rating}/5 | "
            f"🏷️ {r.tags}\n"
            f"  📐 Similarity: {r.similarity:.2f} | 🔑 Chunk: {r.chunk_id}"
        )
    return "\n\n".join(lines)


def query_pipeline(
    question: str,
    top_k: int,
    collection_name: str,
    professor_filter: str,
    department_filter: str,
    min_rating_filter: float,
) -> tuple[str, str, str]:
    """
    Main pipeline function called by Gradio.

    Returns:
        (answer, sources_text, chunks_debug_text)
    """
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    # Build filter
    where = build_where_clause(professor_filter, department_filter, min_rating_filter)

    # Retrieve
    try:
        results = retrieve(
            query=question,
            top_k=int(top_k),
            collection_name=collection_name,
            where=where,
        )
    except Exception as e:
        error_msg = (
            f"Retrieval error: {str(e)}\n\n"
            "Make sure you have run:\n"
            "  python -m src.ingest\n"
            "  python -m src.clean\n"
            "  python -m src.chunk\n"
            "  python -m src.embed"
        )
        return error_msg, "", ""

    # Generate
    try:
        answer = generate_answer(question, results)
    except Exception as e:
        return f"Generation error: {str(e)}", format_sources_for_display(results), ""

    sources_text = format_sources_for_display(results)
    chunks_text = format_chunks_for_display(results)

    return answer, sources_text, chunks_text


# ── Build Gradio Interface ────────────────────────────────────────────────────

with gr.Blocks(
    title="The Unofficial Guide — Denison Professor Reviews",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("# 📚 The Unofficial Guide")
    gr.Markdown(DOMAIN_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Ask a question about Denison professors",
                placeholder="e.g. Which CS professor is best for beginners?",
                lines=2,
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Ask", variant="primary", size="lg")

    # ── Filters (stretch feature) ─────────────────────────────────────────────
    with gr.Accordion("🔍 Filters & Settings (optional)", open=False):
        gr.Markdown(
            "*Narrow results by professor, department, or minimum rating. "
            "Leave blank for unfiltered search.*"
        )
        with gr.Row():
            professor_filter = gr.Textbox(
                label="Filter by Professor Name",
                placeholder='e.g. Dr. Sarah Chen',
                scale=2,
            )
            department_filter = gr.Textbox(
                label="Filter by Department",
                placeholder='e.g. Computer Science',
                scale=2,
            )
            min_rating_filter = gr.Slider(
                label="Min Star Rating",
                minimum=1.0,
                maximum=5.0,
                step=0.5,
                value=1.0,
                scale=1,
            )
        with gr.Row():
            top_k_slider = gr.Slider(
                label="Number of chunks to retrieve (top_k)",
                minimum=1,
                maximum=10,
                step=1,
                value=5,
                scale=2,
            )
            collection_selector = gr.Dropdown(
                label="Chunking Strategy (stretch feature)",
                choices=["review_level", "professor_level"],
                value="review_level",
                scale=1,
                info="review_level = one chunk per review (default). professor_level = all reviews per professor combined.",
            )

    # ── Outputs ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer")
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="Retrieved Sources")

    with gr.Accordion("🔬 Debug: Retrieved Chunks", open=False):
        chunks_output = gr.Textbox(
            label="Raw retrieved chunks (for debugging)",
            lines=15,
            interactive=False,
        )

    # ── Example questions ──────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["What do students think about Sarah Wolff's teaching style?"],
            ["Which Math professor gives the best feedback and is most accessible?"],
            ["Is Ashwin Lall a good professor for Computer Science courses?"],
            ["What is DA101 like — is it a lot of work?"],
            ["Which professors at Denison are known for being tough graders?"],
            ["What restaurants are near Denison?"],  # Out-of-scope example
        ],
        inputs=[question_input],
        label="Example Questions (click to load)",
    )

    # ── Wire up the button ─────────────────────────────────────────────────────
    submit_btn.click(
        fn=query_pipeline,
        inputs=[
            question_input,
            top_k_slider,
            collection_selector,
            professor_filter,
            department_filter,
            min_rating_filter,
        ],
        outputs=[answer_output, sources_output, chunks_output],
    )

    # Also trigger on Enter in the question box
    question_input.submit(
        fn=query_pipeline,
        inputs=[
            question_input,
            top_k_slider,
            collection_selector,
            professor_filter,
            department_filter,
            min_rating_filter,
        ],
        outputs=[answer_output, sources_output, chunks_output],
    )


if __name__ == "__main__":
    demo.launch(share=False)
