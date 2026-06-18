"""BGE-M3 embedding helpers via sentence-transformers."""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import MODEL_NAME, QUERY_PREFIX

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    """Load the BGE-M3 model once and reuse across calls."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        device = _model.device
        logger.info("Model loaded on device: %s", device)
    return _model


def encode_documents(texts: list[str]) -> np.ndarray:
    """
    Encode document texts into L2-normalized vectors.

    Normalization enables cosine similarity via FAISS IndexFlatIP.
    """
    model = load_model()
    logger.info("Encoding %d documents...", len(texts))
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info("Document encoding complete. Shape: %s", vectors.shape)
    return vectors.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a search query with the BGE retrieval prefix.

    Queries use an instruction prefix (asymmetric retrieval); documents do not.
    """
    model = load_model()
    prefixed = QUERY_PREFIX + query
    vector = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.astype(np.float32)
