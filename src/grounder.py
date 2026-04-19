# grounder.py
# Responsibility: construct a grounding prompt, call the LLM API,
# return the answer with source attribution and a drift flag.

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Minimum overlap ratio between answer and retrieved chunks
# below which we flag the response as potentially drifting.
# This threshold is a design decision — not a ground truth.
# It is intentionally conservative (low) to avoid false positives.
DRIFT_THRESHOLD = 0.15


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Construct the grounding prompt.

    The prompt does three things:
    1. Provides the retrieved passages as the only permitted knowledge source
    2. Explicitly instructs the model to abstain if the answer is not present
    3. Asks the model to indicate which passage supported its answer

    The instruction to cite supporting passages is the traceability mechanism.
    It is imperfect — the model sometimes cites the wrong passage or none —
    but the failure pattern itself is a research observation.

    Parameters
    ----------
    question : str
        The user's question.
    chunks : list of dict
        Retrieved chunks from retriever.search(), each with "text" and "source".

    Returns
    -------
    str
        The fully constructed prompt string.
    """

    # Format retrieved passages with numbering for citation reference
    passages = ""
    for i, chunk in enumerate(chunks):
        passages += f"[Passage {i+1} — Source: {chunk['source']}]\n"
        passages += chunk["text"].strip()
        passages += "\n\n"

    prompt = f"""You are a precise and faithful assistant. Answer the question below using ONLY the provided passages.

Rules:
- If the answer is present in the passages, answer faithfully and cite which passage supported your answer.
- If the answer is not present in the passages, respond with exactly: "This is not covered in the provided passages."
- Do not use any knowledge beyond what is written in the passages below.
- Do not infer, speculate, or generalise beyond the passage content.

Passages:
{passages}

Question: {question}

Answer:"""

    return prompt


def call_llm(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Call the Gemini API with the grounding prompt.

    Uses gemini-2.5-flash-lite — fast, free tier, and capable enough
    for grounded single-domain question answering.

    temperature=0.1 keeps generation near-deterministic for
    reproducibility across probe runs.
    """

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=max_new_tokens,
        )
    )

    response = model.generate_content(prompt)
    return response.text.strip()

def detect_drift(answer: str, chunks: list[dict]) -> dict:
    """
    Estimate whether the model's answer drifted beyond the retrieved context.

    Method: compute the proportion of content words in the answer that also
    appear in the retrieved chunks. Low overlap suggests the model drew on
    parametric knowledge rather than the provided passages.

    This is a lexical heuristic — not a semantic measure. It will:
    - Miss drift where the model paraphrases faithfully but differently
    - Flag legitimate answers that use technical vocabulary not in chunks
    These limitations are documented in the research observations.

    Parameters
    ----------
    answer : str
        The model's generated answer.
    chunks : list of dict
        The retrieved chunks used to ground the answer.

    Returns
    -------
    dict with keys:
        "overlap_ratio" : float — proportion of answer words found in chunks
        "drift_flagged" : bool — True if overlap is below DRIFT_THRESHOLD
        "abstained"     : bool — True if the model correctly abstained
    """

    abstained = "not covered in the provided passages" in answer.lower()

    if abstained:
        return {
            "overlap_ratio": 1.0,
            "drift_flagged": False,
            "abstained": True
        }

    # Build a vocabulary of words present in the retrieved chunks
    # Lowercase and split — no stemming, keeping it simple and transparent
    chunk_words = set()
    for chunk in chunks:
        chunk_words.update(chunk["text"].lower().split())

    # Filter answer to content words only — remove short function words
    # that appear everywhere and dilute the signal
    answer_words = [
        w for w in answer.lower().split()
        if len(w) > 3
    ]

    if not answer_words:
        return {
            "overlap_ratio": 0.0,
            "drift_flagged": True,
            "abstained": False
        }

    overlap = sum(1 for w in answer_words if w in chunk_words)
    overlap_ratio = overlap / len(answer_words)

    return {
        "overlap_ratio": round(overlap_ratio, 3),
        "drift_flagged": overlap_ratio < DRIFT_THRESHOLD,
        "abstained": False
    }


def ground_and_answer(question: str, chunks: list[dict]) -> dict:
    """
    Full grounding pipeline for a single question.

    Parameters
    ----------
    question : str
        The user's question.
    chunks : list of dict
        Retrieved chunks from retriever.search().

    Returns
    -------
    dict with keys:
        "question"    : the original question
        "answer"      : the model's generated answer
        "sources"     : list of source filenames and chunk_ids used
        "drift"       : output of detect_drift()
        "prompt"      : the full prompt sent to the model (for inspection)
    """

    prompt = build_prompt(question, chunks)
    answer = call_llm(prompt)
    drift = detect_drift(answer, chunks)

    sources = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c.get("score")}
        for c in chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "drift": drift,
        "prompt": prompt
    }