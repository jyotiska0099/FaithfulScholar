# FaithfulScholar

> Can an LLM answer questions about an expert's work — faithfully, traceably, and without drifting beyond what the expert actually said?

FaithfulScholar is a research probe, not a product. It grounds a large language model
in a personal research corpus — papers, thesis chapters, and notes — and enforces
answer generation strictly from retrieved source passages. Every answer is attributed
to the exact passage it came from. Every failure to stay grounded is logged.

The system is built to be deliberately broken: its failure modes — hallucination on
out-of-corpus queries, retrieval degradation under embedding compression, and grounding
drift despite explicit constraints — are the research observations, not bugs to fix.
This directly motivates an open problem in NLP: does knowledge distillation preserve
grounding, or only fluency?

**Live demo:** `[HuggingFace Spaces link — coming soon]`
**Research note:** `[one-page analysis PDF link — coming soon]`

---

## Quick Start
```bash
git clone https://github.com/jyotiska/faithfulscholar
cd faithfulscholar
pip install -r requirements.txt
python app.py
```

Add your own documents to `/corpus` as `.txt` or `.pdf` files.
The system will chunk, embed, and index them automatically on first run.

---

## The Problem This Explores

Current LLMs are unsuitable as faithful representations of a specific expert's
knowledge for three reasons:

- **Hallucination** — they generate fluent answers not grounded in the expert's
  actual corpus
- **Opacity** — there is no traceable link between an answer and its source
- **Compression risk** — when distilled to smaller models for deployment, it is
  unclear whether grounding is preserved or only surface fluency

Retrieval-Augmented Generation (RAG) is the standard engineering response to this
problem. FaithfulScholar treats RAG not as a solution but as a controlled experimental
setting — a way to systematically study where and why grounding fails, and what
happens to faithfulness when the retrieval component is compressed.

---

## System Architecture
```
Personal Corpus (papers, thesis, notes)
           │
           ▼
       Chunking
  (paragraph-level, ~200 tokens)
           │
           ▼
  Sentence Transformer
  (all-MiniLM-L6-v2)
           │
           ▼
    Dense Vectors
           │
           ▼
      FAISS Index  ◄──── Query Vector ◄──── Sentence Transformer ◄──── Question
           │
           ▼
     Top-k Chunks
     (k=3 default)
           │
           ▼
  LLM + Grounding Prompt
  (HuggingFace Inference API)
           │
           ▼
  Answer + Source Attribution + Faithfulness Log
```

---

## Design Decisions and Rationale

### Chunking strategy
Documents are chunked at paragraph boundaries rather than fixed token counts.
Fixed-size chunking is simpler but routinely cuts a single idea across two chunks,
degrading retrieval coherence. Paragraph-level chunking preserves semantic units
at the cost of uneven chunk sizes — an acceptable tradeoff for a research corpus
where coherence matters more than uniformity.

### Embedding model
`all-MiniLM-L6-v2` is chosen as the full-size baseline — compact enough to run
locally without a GPU, strong enough on semantic similarity benchmarks to serve
as a credible retrieval layer. A quantised variant of the same model serves as
the compression probe in Stage 4.

### Grounding constraint
The grounding constraint is enforced through prompting only — the model receives
the retrieved passages and an explicit instruction to answer solely from them.
This is architecturally weak: the model is not prevented from accessing its
parametric knowledge, only instructed not to. This weakness is intentional.
Documenting when and how the model violates the constraint under this soft
enforcement is one of the core research observations.

### k=3 retrieval default
Retrieving three passages balances context richness against dilution. With k=1,
relevant context is frequently missed. With k=5 or more, irrelevant passages
enter the context window and increase the probability of grounding drift.
k=3 is the starting point; the effect of varying k is logged as part of the
faithfulness analysis.

---

## Research Probes

### Probe 1 — In-corpus faithfulness
Questions whose answers are directly present in the corpus.
Expected behaviour: correct retrieval, faithful answer, clear source attribution.
Logged: retrieval rank of the correct chunk, answer overlap with source passage.

### Probe 2 — Out-of-corpus hallucination
Questions whose answers are not present in the corpus.
Expected behaviour: the model abstains — "this is not covered in the provided
passages."
Logged: rate of correct abstention vs. hallucination, linguistic markers of
drift (hedging phrases, confident assertions unsupported by retrieved context).

### Probe 3 — Compression faithfulness
The full `all-MiniLM-L6-v2` embedding model is replaced with its INT8 quantised
variant. Retrieval quality (recall@k) is measured against the full-model baseline.
Downstream faithfulness of answers is then compared across both settings.
Research question: does embedding compression degrade grounding, and if so, at
what rate relative to model size reduction?

---

## Repository Structure
```
faithfulscholar/
│
├── corpus/                  # Your documents go here (.txt or .pdf)
├── index/                   # FAISS index stored after first run
│
├── src/
│   ├── chunker.py           # Document loading and paragraph chunking
│   ├── embedder.py          # Sentence transformer wrapper, full and quantised
│   ├── retriever.py         # FAISS indexing and nearest-neighbour search
│   ├── grounder.py          # LLM call with grounding prompt and attribution
│   └── probe.py             # Faithfulness logging and probe result export
│
├── app.py                   # Gradio interface
├── analysis/
│   └── faithfulness_report.pdf   # One-page research observation writeup
│
├── requirements.txt
└── README.md
```

---

## Observations

## Observations

### Probe 1 — In-corpus faithfulness (full model)
4 out of 5 in-corpus questions answered faithfully with average lexical overlap
of 0.816. The system correctly attributed answers to source passages in all
answered cases. One abstention was recorded for the query "What neural
architecture did you use for stress detection?" — a retrieval failure, not a
knowledge gap. The answer existed in the corpus but the linguistic distance
between the query phrasing and the methods section language was sufficient to
prevent correct retrieval (top score: 0.555). Rephrasing the query to "What is
TinyTCN?" retrieved the correct passage successfully, confirming the information
was present but not surfaced by the original phrasing.

### Probe 2 — Out-of-corpus hallucination (full model)
5 out of 5 out-of-corpus questions correctly abstained. The soft grounding
constraint — enforced through prompting only, with no architectural prevention
of parametric knowledge access — held completely for a capable instruction-tuned
model (Gemini 2.0 Flash) on questions entirely outside the research domain.
The drift detector recorded overlap ratio of 1.0 for all abstentions, as the
abstention phrase itself overlaps with the known vocabulary.

### Probe 3 — Compression faithfulness (compressed model)
Average lexical overlap dropped from 0.816 (full model) to 0.745 (compressed
model) on the same in-corpus question set — a 7.1% degradation in retrieval
faithfulness under embedding compression. The compressed model (L3, half the
layers of the full L6 model) produced 0 abstentions versus 1 in Probe 1,
suggesting it retrieved different chunks for the previously abstained question.
Whether this represents improved retrieval or reduced conservatism requires
further investigation.

### Key findings
- Retrieval failure is a more common faithfulness failure mode than hallucination
  under a capable model with explicit grounding constraints.
- Query phrasing sensitivity is a significant vulnerability — the same information
  retrieves correctly under one phrasing and fails under another.
- Embedding compression produces measurable retrieval degradation (7.1%) without
  triggering the lexical drift detector, suggesting faithfulness loss can be
  invisible to surface-level metrics.
- The lexical drift detector did not trigger across 15 queries, indicating the
  0.15 threshold is too conservative for a capable model — or that Gemini 2.0
  Flash is sufficiently instruction-following to stay within provided context
  without detectable lexical drift.

### Open questions motivating doctoral research
- Does the grounding constraint hold under weaker models or more ambiguous queries?
- Can retrieval be made robust to query phrasing variation without requiring
  the user to know the document's exact terminology?
- When a grounded model is distilled, does the distilled model inherit the
  retrieval sensitivity or does compression change the failure mode?

---

## Motivation and Connection to Ongoing Research

This project is a preliminary exploration of a problem I am pursuing: if we ground an LLM in an expert's knowledge and then compress
that grounded model for edge deployment, does the distilled model retain the
grounding or only the fluency? FaithfulScholar operationalises the first half
of that question — grounding and its failure modes — in a controlled,
single-domain setting before the problem is approached at research scale.

---

## Author

**Jyotiska Bharadwaj**
M.Tech. Software Engineering, Delhi Technological University
[jyotiskabharadwaj@gmail.com](mailto:jyotiskabharadwaj@gmail.com) |
[LinkedIn](https://linkedin.com/in/jyotiska-bharadwaj) |
[ORCID](https://orcid.org/0009-0002-2635-6796)