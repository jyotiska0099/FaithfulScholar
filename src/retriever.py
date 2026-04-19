# retriever.py
# Responsibility: build and search a FAISS index over chunk embeddings.
# Maps FAISS integer indices back to chunk dictionaries for source attribution.

import faiss
import numpy as np
import pickle
import os


# Paths where the index and chunk list are persisted after first build.
# Stored separately because FAISS saves its own binary format,
# while chunks are plain Python objects serialised with pickle.
INDEX_PATH = "index/faiss.index"
CHUNKS_PATH = "index/chunks.pkl"


class Retriever:
    """
    Builds, saves, loads, and searches a FAISS flat index.

    Uses IndexFlatIP — inner product similarity — over normalised vectors,
    which is equivalent to cosine similarity. Cosine similarity is the
    standard metric for sentence transformer embeddings because the
    magnitude of the vector is not meaningful — only its direction is.
    """

    def __init__(self):
        self.index = None
        # chunks is stored alongside the index so that integer positions
        # returned by FAISS can be mapped back to text and source metadata.
        self.chunks = None

    def build(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Build a FAISS index from embeddings and save both index and chunks to disk.

        Parameters
        ----------
        chunks : list of dict
            The chunk dictionaries from chunker.load_corpus().
            Must be in the same order as embeddings — position i in chunks
            corresponds to row i in embeddings.

        embeddings : np.ndarray of shape (num_chunks, embedding_dim)
            Output of embedder.embed_chunks().
        """

        embedding_dim = embeddings.shape[1]

        # Normalise all vectors to unit length before adding to the index.
        # After normalisation, inner product == cosine similarity.
        # faiss.normalize_L2 modifies the array in place.
        faiss.normalize_L2(embeddings)

        # IndexFlatIP performs exact search using inner product.
        # "Flat" means no approximation — every vector is compared to the query.
        # For a research corpus of hundreds to low thousands of chunks,
        # exact search is fast enough and preferable — approximation would
        # introduce retrieval errors that confound the faithfulness probe.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"FAISS index built. Vectors stored: {self.index.ntotal}")

        # Persist to disk so subsequent runs skip the embedding step
        os.makedirs("index", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Index saved to {INDEX_PATH}")
        print(f"Chunks saved to {CHUNKS_PATH}")

    def load(self) -> bool:
        """
        Load a previously built index and chunk list from disk.

        Returns
        -------
        bool
            True if loading succeeded, False if no saved index exists.
            The caller uses this to decide whether to build or load.
        """

        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            return False

        self.index = faiss.read_index(INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Index loaded. Vectors: {self.index.ntotal}, Chunks: {len(self.chunks)}")
        return True

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[dict]:
        """
        Search the index for the k most similar chunks to the query vector.

        Parameters
        ----------
        query_vector : np.ndarray of shape (1, embedding_dim)
            Output of embedder.embed_query().

        k : int
            Number of chunks to retrieve. Default is 3.
            The effect of varying k is logged by probe.py.

        Returns
        -------
        list of dict
            The top-k chunk dictionaries, ordered by similarity descending.
            Each dict has "text", "source", "chunk_id", and an added
            "score" key with the inner product similarity value.
        """

        if self.index is None or self.chunks is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        # Normalise the query vector to unit length — same transformation
        # applied to the corpus embeddings at index build time.
        # Skipping this would make similarity scores inconsistent.
        faiss.normalize_L2(query_vector)

        # search() returns two arrays of shape (1, k):
        # scores — inner product similarity values, higher is more similar
        # indices — integer positions in the FAISS index
        scores, indices = self.index.search(query_vector, k)

        # Flatten from (1, k) to (k,) — we only have one query at a time
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            # FAISS returns -1 for indices when fewer than k vectors exist
            # in the index — guard against this to avoid an index error
            if idx == -1:
                continue

            # Copy the chunk dict and attach the similarity score
            # Copy is important — we don't want to mutate the stored chunks
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results