"""FAISS index and metadata persistence helpers."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP index from L2-normalized vectors.

    Inner product on unit vectors equals cosine similarity.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    """Persist the FAISS index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("FAISS index saved to %s", path)


def load_index(path: Path) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}. Run embed_and_index.py first."
        )
    index = faiss.read_index(str(path))
    logger.info("FAISS index loaded: %d vectors from %s", index.ntotal, path)
    return index


def save_metadata(articles: list[dict], path: Path) -> None:
    """
    Save article metadata as JSON.

    List order must match FAISS row IDs (index i -> metadata[i]).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info("Metadata saved: %d articles to %s", len(articles), path)


def load_metadata(path: Path) -> list[dict]:
    """Load article metadata from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}. Run embed_and_index.py first."
        )
    with path.open(encoding="utf-8") as f:
        articles = json.load(f)
    logger.info("Metadata loaded: %d articles from %s", len(articles), path)
    return articles


def search(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the top-k most similar vectors.

    Returns (scores, indices) arrays of shape (1, k).
    """
    scores, indices = index.search(query_vector, k)
    return scores, indices
