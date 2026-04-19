# embedder.py
# Responsibility: convert chunk text into dense vectors using a sentence transformer.
# Produces embeddings only — does not store or index them.

from sentence_transformers import SentenceTransformer
import numpy as np


# Model identifiers as constants — not hardcoded strings scattered through the code.
# If you want to swap models later, you change it in one place only.
FULL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUANTISED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Note: true INT8 quantisation of sentence transformers requires ONNX runtime.
# For this project, we approximate the compression probe by using a genuinely
# smaller model as the compressed variant — this is honest and researchable.
COMPRESSED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
# L3 has half the layers of L6 — fewer parameters, faster, lower quality.
# This gives us a real compression comparison without ONNX overhead.


class Embedder:
    """
    Wraps a SentenceTransformer model and embeds chunks.

    The use_compressed flag switches between the full model and the
    compressed variant — allowing the compression probe to swap the
    embedding layer without changing anything else in the pipeline.
    """

    def __init__(self, use_compressed: bool = False):
        """
        Parameters
        ----------
        use_compressed : bool
            If False, load the full L6 model (baseline).
            If True, load the smaller L3 model (compression probe).
        """

        model_name = COMPRESSED_MODEL if use_compressed else FULL_MODEL
        self.model_name = model_name
        self.use_compressed = use_compressed

        print(f"Loading embedding model: {model_name}")
        # SentenceTransformer downloads the model on first use and caches it.
        # Subsequent loads use the cache — no repeated downloads.
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """
        Embed a list of chunk dictionaries.

        Parameters
        ----------
        chunks : list of dict
            Output of chunker.load_corpus() — each dict has a "text" key.

        Returns
        -------
        np.ndarray of shape (num_chunks, embedding_dim)
            One vector per chunk, in the same order as the input list.
            Preserving order is critical — the retriever maps vector index
            back to chunk index to recover source metadata.
        """

        # Extract just the text from each chunk dictionary
        texts = [chunk["text"] for chunk in chunks]

        # encode() runs the transformer over all texts.
        # show_progress_bar=True is useful — embedding a full corpus takes
        # 10-30 seconds and you want to see it progressing, not hang silently.
        # convert_to_numpy=True returns an ndarray, which FAISS requires.
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # embeddings.shape will be (len(texts), 384) for L6
        # and (len(texts), 384) for L3 — same dimension, different quality.
        # This matters: FAISS index dimension must match embedding dimension.
        # Since both models here output 384-dim vectors, the index is reusable.
        # If you swap to a different model family, you must rebuild the index.
        print(f"Embedded {len(texts)} chunks. Shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.

        Parameters
        ----------
        query : str
            The user's question.

        Returns
        -------
        np.ndarray of shape (1, embedding_dim)
            FAISS expects a 2D array even for a single query — hence the
            reshape. A 1D vector would cause a silent shape error downstream.
        """

        vector = self.model.encode(query, convert_to_numpy=True)

        # reshape from (384,) to (1, 384)
        return vector.reshape(1, -1)