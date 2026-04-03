# File Name: chunker.py
# Responsibility: load a PDF, extract text, split into paragraph-level chunks.
# Nothing else. No embedding, no indexing, no API calls.

import fitz  # pymupdf — for PDF text extraction
import os


def clean_text(text: str) -> str:
    """
    Light cleaning of raw PDF-extracted text.

    PDFs encode text visually, not semantically. Extraction artifacts include:
    - Hyphenated line breaks: "knowl-\nedge" should be "knowledge"
    - Excessive whitespace from column layouts or spacing
    - Isolated newlines within a paragraph that are not paragraph boundaries

    We do not do aggressive cleaning — we want to preserve the author's
    phrasing as faithfully as possible, since faithfulness to source text
    is the research question this system probes.
    """

    import re

    # Rejoin words hyphenated across line breaks
    # Pattern: a letter, a hyphen, a newline, a lowercase letter
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Replace single newlines within a paragraph with a space
    # Double newlines (paragraph boundaries) are preserved
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Normalize multiple spaces to a single space
    text = re.sub(r' +', ' ', text)

    # Normalize more than two consecutive newlines to exactly two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def chunk_document(filepath: str, min_chunk_length: int = 100) -> list[dict]:
    """
    Load a PDF and return a list of paragraph-level chunks with metadata.

    Parameters
    ----------
    filepath : str
        Path to the PDF file.
    min_chunk_length : int
        Minimum character length for a chunk to be kept.
        Chunks shorter than this are likely headers, captions, or page
        numbers — not meaningful knowledge units.

    Returns
    -------
    list of dict, each with keys:
        "text"     : the chunk content (str)
        "source"   : the filename, not the full path (str)
        "chunk_id" : integer index of this chunk within this document (int)
    """

    # Verify the file exists before attempting to open it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at: {filepath}")

    # Extract filename from path for metadata
    # We store only the filename, not the full path, for portability
    filename = os.path.basename(filepath)

    # Open the PDF with pymupdf
    # fitz.open returns a Document object — iterable over pages
    doc = fitz.open(filepath)

    # Extract text from every page and join with double newline
    # This preserves page-level paragraph boundaries in most PDFs
    full_text = "\n\n".join(page.get_text() for page in doc)

    doc.close()

    # Clean the extracted text
    full_text = clean_text(full_text)

    # Split on double newlines to get paragraph-level blocks
    raw_chunks = full_text.split("\n\n")

    # Filter and package chunks
    chunks = []
    chunk_id = 0

    for block in raw_chunks:
        block = block.strip()

        # Discard blocks that are too short to be meaningful
        # min_chunk_length is a parameter so you can tune it per corpus
        if len(block) < min_chunk_length:
            continue

        chunks.append({
            "text": block,
            "source": filename,
            "chunk_id": chunk_id
        })

        chunk_id += 1

    return chunks


def load_corpus(corpus_dir: str) -> list[dict]:
    """
    Load and chunk all PDF files in a directory.

    Parameters
    ----------
    corpus_dir : str
        Path to the folder containing your PDF documents.

    Returns
    -------
    list of dict — all chunks from all documents, with source metadata.
    chunk_id is unique within each document, not globally.
    """

    all_chunks = []

    # List only PDF files — ignore hidden files, subdirectories, etc.
    pdf_files = [
        f for f in os.listdir(corpus_dir)
        if f.endswith(".pdf") and not f.startswith(".")
    ]

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {corpus_dir}")

    for filename in sorted(pdf_files):
        filepath = os.path.join(corpus_dir, filename)
        print(f"Chunking: {filename}")

        doc_chunks = chunk_document(filepath)
        all_chunks.extend(doc_chunks)

        print(f"  → {len(doc_chunks)} chunks extracted")

    print(f"\nTotal chunks across corpus: {len(all_chunks)}")
    return all_chunks