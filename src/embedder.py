# embedder.py
# Responsibility: convert chunk text into dense vectors.
# Three modes:
#   use_compressed=False, use_onnx=False  → full PyTorch L6 (baseline)
#   use_compressed=True,  use_onnx=False  → smaller PyTorch L3 (architecture compression)
#   use_compressed=False, use_onnx=True   → ONNX INT8 L6 (precision compression)
#
# All three modes expose the same interface:
#   embed_chunks(chunks) → np.ndarray of shape (n, 384)
#   embed_query(query)   → np.ndarray of shape (1, 384)
#
# The rest of the pipeline never needs to know which mode is active.

from sentence_transformers import SentenceTransformer
import numpy as np
import os


FULL_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
COMPRESSED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
ONNX_DIR         = "models/onnx"
ONNX_FP32_PATH   = os.path.join(ONNX_DIR, "model.onnx")
ONNX_INT8_PATH   = os.path.join(ONNX_DIR, "model_int8.onnx")


class Embedder:
    """
    Wraps sentence embedding under three interchangeable backends.

    The use_onnx flag takes priority over use_compressed — if use_onnx=True,
    the ONNX INT8 backend is used regardless of use_compressed.
    """

    def __init__(self, use_compressed: bool = False, use_onnx: bool = False):
        """
        Parameters
        ----------
        use_compressed : bool
            If True and use_onnx is False, load the smaller L3 PyTorch model.
        use_onnx : bool
            If True, load the ONNX INT8 model via ONNX Runtime.
            Takes priority over use_compressed.
        """

        self.use_onnx       = use_onnx
        self.use_compressed = use_compressed

        if use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch()

    def _init_pytorch(self):
        """Initialise the PyTorch sentence transformer backend."""

        model_name   = COMPRESSED_MODEL if self.use_compressed else FULL_MODEL
        self.model_name = model_name
        self.model      = None  # ONNX path uses different attributes

        print(f"Loading PyTorch embedding model: {model_name}")
        self._st_model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: "
              f"{self._st_model.get_sentence_embedding_dimension()}")

    def _init_onnx(self):
        """
        Initialise the ONNX Runtime backend for INT8 inference.

        ONNX Runtime does not use the SentenceTransformer abstraction.
        Instead we:
          1. Load the tokenizer separately (saved alongside the ONNX model)
          2. Create an InferenceSession over the INT8 .onnx file
          3. Implement mean pooling and normalisation manually

        Mean pooling: average the token embeddings across the sequence length
        dimension, weighted by the attention mask so padding tokens are excluded.

        Normalisation: divide each vector by its L2 norm so that inner product
        equals cosine similarity — the same transformation applied in the
        PyTorch path via faiss.normalize_L2.
        """

        from transformers import AutoTokenizer
        import onnxruntime as ort

        if not os.path.exists(ONNX_INT8_PATH):
            raise FileNotFoundError(
                f"ONNX INT8 model not found at {ONNX_INT8_PATH}. "
                f"Run scripts/export_onnx.py first."
            )

        self.model_name  = f"ONNX-INT8:{ONNX_INT8_PATH}"
        self._st_model   = None  # PyTorch path uses this attribute

        print(f"Loading ONNX INT8 model from {ONNX_INT8_PATH}")

        self._tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)

        # SessionOptions can control thread count and optimisation level.
        # Defaults are fine for a research probe on a laptop CPU.
        self._session = ort.InferenceSession(
            ONNX_INT8_PATH,
            providers=["CPUExecutionProvider"]
        )

        print("ONNX INT8 model loaded. Embedding dimension: 384")

    # ── Shared interface ───────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """
        Embed a list of chunk dictionaries.

        Returns np.ndarray of shape (num_chunks, 384).
        Order is preserved — position i in chunks corresponds to row i
        in the returned array.
        """

        texts = [chunk["text"] for chunk in chunks]

        if self.use_onnx:
            embeddings = self._encode_onnx(texts)
        else:
            embeddings = self._st_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        print(f"Embedded {len(texts)} chunks. Shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns np.ndarray of shape (1, 384).
        FAISS requires a 2D array even for a single query.
        """

        if self.use_onnx:
            vector = self._encode_onnx([query])[0]
        else:
            vector = self._st_model.encode(query, convert_to_numpy=True)

        return vector.reshape(1, -1)

    # ── ONNX inference internals ───────────────────────────────────────────────

    def _encode_onnx(self, texts: list[str]) -> np.ndarray:
        """
        Tokenise texts and run inference through the ONNX Runtime session.

        Steps:
          1. Tokenise with padding and truncation
          2. Run the ONNX session — outputs last_hidden_state of shape
             (batch, seq_len, 384)
          3. Mean pool across seq_len dimension, excluding padding tokens
          4. L2 normalise each vector

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray of shape (len(texts), 384)
        """

        # Tokenise — padding to longest sequence in batch, truncate at 128
        # tokens (MiniLM's effective context length for sentence similarity)
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"   # return numpy arrays directly
        )

        # ONNX Runtime expects a dict of input_name → numpy array
        # Input names must match what was specified at export time
        ort_inputs = {
            "input_ids":      encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        # token_type_ids is optional for MiniLM but include if present
        if "token_type_ids" in encoded:
            ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        # Run inference — returns list of outputs in order defined at export
        # For feature extraction, first output is last_hidden_state:
        # shape (batch_size, seq_len, hidden_dim)
        ort_outputs  = self._session.run(None, ort_inputs)
        hidden_state = ort_outputs[0]  # (batch, seq_len, 384)

        # ── Mean pooling ───────────────────────────────────────────────────────
        # Expand attention mask to match hidden state dimensions for masking
        attention_mask = encoded["attention_mask"]                  # (batch, seq_len)
        mask_expanded  = attention_mask[:, :, np.newaxis].astype(   # (batch, seq_len, 1)
            np.float32
        )

        # Zero out padding token embeddings, then sum across seq_len
        sum_embeddings = np.sum(hidden_state * mask_expanded, axis=1)  # (batch, 384)

        # Divide by number of non-padding tokens per sequence
        # Clamp to minimum 1e-9 to avoid division by zero on empty sequences
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        embeddings = sum_embeddings / sum_mask   # (batch, 384)

        # ── L2 normalisation ───────────────────────────────────────────────────
        norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms      = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)