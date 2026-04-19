# probe.py
# Responsibility: run structured faithfulness probes, log all results to disk.
# Orchestrates the full pipeline across three experimental conditions.

import json
import os
from datetime import datetime
from src.chunker import load_corpus
from src.embedder import Embedder
from src.retriever import Retriever
from src.grounder import ground_and_answer


# Output path for probe logs
LOG_DIR = "analysis"
LOG_FILE = os.path.join(LOG_DIR, "probe_log.json")


# ── Probe question sets ────────────────────────────────────────────────────────
# These are the questions you design before running the probes.
# In-corpus questions should be answerable directly from your papers.
# Out-of-corpus questions should be entirely outside your research domain.
# Keep both sets small but representative — 5 questions each is sufficient.

IN_CORPUS_QUESTIONS = [
    "What neural architecture did you use for stress detection?",
    "Which dataset did you use to train and evaluate your model?",
    "What optimisation technique did you apply to reduce model size?",
    "What physiological signals did your system analyse?",
    "What was the accuracy of your best performing model?",
]

OUT_OF_CORPUS_QUESTIONS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is the GDP of India in 2024?",
    "Who wrote the novel Middlemarch?",
    "What is the boiling point of ethanol?",
]


# ── Core probe runner ──────────────────────────────────────────────────────────

def run_probe(
    questions: list[str],
    probe_name: str,
    embedder: Embedder,
    retriever: Retriever,
    k: int = 3
) -> list[dict]:
    """
    Run a set of questions through the full pipeline and collect results.

    Parameters
    ----------
    questions : list of str
        The probe question set.
    probe_name : str
        Label for this probe — stored in each log record for filtering later.
    embedder : Embedder
        The active embedder instance — full or compressed.
    retriever : Retriever
        A built or loaded retriever instance.
    k : int
        Number of chunks to retrieve per question.

    Returns
    -------
    list of dict
        One record per question, containing all pipeline outputs.
    """

    records = []

    for question in questions:
        print(f"\n[{probe_name}] Q: {question}")

        # Embed the query using the active embedder
        query_vector = embedder.embed_query(question)

        # Retrieve top-k chunks
        retrieved_chunks = retriever.search(query_vector, k=k)

        # Ground and answer
        result = ground_and_answer(question, retrieved_chunks)

        # Build the log record
        record = {
            "probe": probe_name,
            "embedding_model": embedder.model_name,
            "k": k,
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "drift": result["drift"],
            "retrieved_chunks": [
                {
                    "text": c["text"][:300],  # truncate for log readability
                    "source": c["source"],
                    "chunk_id": c["chunk_id"],
                    "score": c.get("score")
                }
                for c in retrieved_chunks
            ],
            "timestamp": datetime.now().isoformat()
        }

        records.append(record)

        # Print a summary to terminal during the run
        print(f"  Answer: {result['answer'][:150]}...")
        print(f"  Drift flagged: {result['drift']['drift_flagged']}")
        print(f"  Overlap ratio: {result['drift']['overlap_ratio']}")
        print(f"  Abstained: {result['drift']['abstained']}")

    return records


# ── Main probe orchestrator ────────────────────────────────────────────────────

def run_all_probes():
    """
    Run all three probes sequentially and save a unified log to disk.

    Probe 1: In-corpus questions, full embedding model
    Probe 2: Out-of-corpus questions, full embedding model
    Probe 3: In-corpus questions, compressed embedding model
             (same questions as Probe 1, different embedder)
             Compare results against Probe 1 to isolate compression effect.
    """

    os.makedirs(LOG_DIR, exist_ok=True)
    all_records = []

    # ── Shared corpus loading ──────────────────────────────────────────────────
    # Load corpus once — shared across all probes
    print("Loading corpus...")
    chunks = load_corpus("corpus")

    # ── Probe 1 & 2: Full embedding model ─────────────────────────────────────
    print("\nInitialising full embedding model...")
    full_embedder = Embedder(use_compressed=False)
    full_embeddings = full_embedder.embed_chunks(chunks)

    full_retriever = Retriever()
    full_retriever.build(chunks, full_embeddings)

    print("\n── Probe 1: In-corpus faithfulness (full model) ──")
    probe1_records = run_probe(
        IN_CORPUS_QUESTIONS,
        probe_name="probe_1_in_corpus_full",
        embedder=full_embedder,
        retriever=full_retriever
    )
    all_records.extend(probe1_records)

    print("\n── Probe 2: Out-of-corpus hallucination (full model) ──")
    probe2_records = run_probe(
        OUT_OF_CORPUS_QUESTIONS,
        probe_name="probe_2_out_of_corpus_full",
        embedder=full_embedder,
        retriever=full_retriever
    )
    all_records.extend(probe2_records)

    # ── Probe 3: Compressed embedding model ───────────────────────────────────
    # Build a separate index using the compressed embedder.
    # The index must be rebuilt — you cannot reuse the full model's index
    # because the vectors were produced by a different model.
    # We save the compressed index to a separate path to avoid overwriting
    # the full model index.
    print("\nInitialising compressed embedding model...")
    compressed_embedder = Embedder(use_compressed=True)
    compressed_embeddings = compressed_embedder.embed_chunks(chunks)

    # Temporarily redirect index paths for the compressed index
    import src.retriever as retriever_module
    original_index_path = retriever_module.INDEX_PATH
    original_chunks_path = retriever_module.CHUNKS_PATH
    retriever_module.INDEX_PATH = "index/faiss_compressed.index"
    retriever_module.CHUNKS_PATH = "index/chunks_compressed.pkl"

    compressed_retriever = Retriever()
    compressed_retriever.build(chunks, compressed_embeddings)

    print("\n── Probe 3: In-corpus faithfulness (compressed model) ──")
    probe3_records = run_probe(
        IN_CORPUS_QUESTIONS,
        probe_name="probe_3_in_corpus_compressed",
        embedder=compressed_embedder,
        retriever=compressed_retriever
    )
    all_records.extend(probe3_records)
    
    # ── Probe 4: ONNX INT8 embedding model ────────────────────────────────────
    # Same questions as Probe 1, same architecture (L6), INT8 quantised weights.
    # Compared against Probe 1: isolates the effect of precision reduction only.
    # Compared against Probe 3: separates architectural compression (L3) from
    # precision compression (INT8) as two distinct degradation mechanisms.
    print("\nInitialising ONNX INT8 embedding model...")
    onnx_embedder = Embedder(use_onnx=True)
    onnx_embeddings = onnx_embedder.embed_chunks(chunks)

    retriever_module.INDEX_PATH  = "index/faiss_onnx.index"
    retriever_module.CHUNKS_PATH = "index/chunks_onnx.pkl"

    onnx_retriever = Retriever()
    onnx_retriever.build(chunks, onnx_embeddings)

    print("\n── Probe 4: In-corpus faithfulness (ONNX INT8) ──")
    probe4_records = run_probe(
        IN_CORPUS_QUESTIONS,
        probe_name="probe_4_in_corpus_onnx_int8",
        embedder=onnx_embedder,
        retriever=onnx_retriever
    )
    all_records.extend(probe4_records)

    # ── Save all records ───────────────────────────────────────────────────────
    with open(LOG_FILE, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nAll probes complete. Log saved to {LOG_FILE}")

    # ── Print summary statistics ───────────────────────────────────────────────
    print_summary(all_records)


def print_summary(records: list[dict]):
    """
    Print a concise summary of probe results to terminal.
    This is not the analysis — it is a quick orientation before
    you read the full log and write the one-page analysis.
    """

    print("\n── Summary ──────────────────────────────────────────────────────")

    probes = {}
    for r in records:
        probes.setdefault(r["probe"], []).append(r)

    for probe_name, probe_records in probes.items():
        total = len(probe_records)
        drifted = sum(1 for r in probe_records if r["drift"]["drift_flagged"])
        abstained = sum(1 for r in probe_records if r["drift"]["abstained"])
        avg_overlap = sum(
            r["drift"]["overlap_ratio"] for r in probe_records
        ) / total

        print(f"\n{probe_name}")
        print(f"  Questions    : {total}")
        print(f"  Drift flagged: {drifted} / {total}")
        print(f"  Abstained    : {abstained} / {total}")
        print(f"  Avg overlap  : {avg_overlap:.3f}")


if __name__ == "__main__":
    run_all_probes()