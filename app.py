# app.py
# Responsibility: Gradio interface for FaithfulScholar.
# Initialises the pipeline once on startup, then serves queries.

import gradio as gr
import json
from src.chunker import load_corpus
from src.embedder import Embedder
from src.retriever import Retriever
from src.grounder import ground_and_answer

# ── Pipeline initialisation ────────────────────────────────────────────────────
# These are initialised once at startup and reused across all queries.
# Gradio runs the interface in a long-lived process — initialising per query
# would reload the embedding model on every request, which takes 10+ seconds.

print("Initialising FaithfulScholar pipeline...")

chunks = load_corpus("corpus")
embedder = Embedder(use_compressed=False)
retriever = Retriever()

if not retriever.load():
    print("No saved index found. Building from corpus...")
    embeddings = embedder.embed_chunks(chunks)
    retriever.build(chunks, embeddings)

print("Pipeline ready.\n")


# ── Core query function ────────────────────────────────────────────────────────

def answer_question(question: str, k: int) -> tuple[str, str, str]:
    """
    Run a question through the full pipeline and return formatted outputs.

    Parameters
    ----------
    question : str
        The user's question from the Gradio interface.
    k : int
        Number of chunks to retrieve — controlled by a Gradio slider.

    Returns
    -------
    tuple of three strings:
        - answer text
        - source attribution formatted for display
        - drift assessment formatted for display
    """

    if not question.strip():
        return "Please enter a question.", "", ""

    # Embed the query and retrieve
    query_vector = embedder.embed_query(question)
    retrieved_chunks = retriever.search(query_vector, k=k)

    # Ground and answer
    result = ground_and_answer(question, retrieved_chunks)

    # ── Format source attribution ──────────────────────────────────────────────
    source_lines = []
    for i, chunk in enumerate(retrieved_chunks):
        source_lines.append(
            f"**Passage {i+1}** — {chunk['source']} "
            f"(chunk {chunk['chunk_id']}, score: {chunk['score']:.3f})\n\n"
            f"{chunk['text'][:400]}..."
        )
    sources_display = "\n\n---\n\n".join(source_lines)

    # ── Format drift assessment ────────────────────────────────────────────────
    drift = result["drift"]

    if drift["abstained"]:
        drift_display = "✅ **Abstained** — model correctly reported the answer is not in the corpus."
    elif drift["drift_flagged"]:
        drift_display = (
            f"⚠️ **Drift flagged** — overlap ratio: {drift['overlap_ratio']:.3f}\n\n"
            f"The answer may draw on knowledge outside the retrieved passages."
        )
    else:
        drift_display = (
            f"✅ **Grounded** — overlap ratio: {drift['overlap_ratio']:.3f}\n\n"
            f"The answer appears faithful to the retrieved passages."
        )

    return result["answer"], sources_display, drift_display


# ── Gradio interface ───────────────────────────────────────────────────────────

with gr.Blocks(title="FaithfulScholar") as demo:

    gr.Markdown("""
    # FaithfulScholar
    **A research probe for faithful, traceable knowledge grounding in LLMs.**

    Ask a question about the research corpus. Every answer is attributed to its
    source passage and assessed for grounding faithfulness.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Question",
                placeholder="What neural architecture did you use for stress detection?",
                lines=2
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Number of passages to retrieve (k)",
                info="Higher k retrieves more context but increases hallucination risk"
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            drift_output = gr.Markdown(label="Faithfulness Assessment")

    answer_output = gr.Textbox(
        label="Answer",
        lines=6,
        interactive=False
    )

    sources_output = gr.Markdown(label="Retrieved Source Passages")

    # ── Event binding ──────────────────────────────────────────────────────────
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output, drift_output]
    )

    # Allow Enter key to submit
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, k_slider],
        outputs=[answer_output, sources_output, drift_output]
    )

    gr.Markdown("""
    ---
    **Research note:** The faithfulness assessment uses a lexical overlap heuristic.
    An overlap ratio below 0.15 flags potential drift. Abstention indicates the model
    correctly identified the answer as outside the corpus.
    [View full probe log](analysis/probe_log.json)
    """)

if __name__ == "__main__":
    demo.launch()