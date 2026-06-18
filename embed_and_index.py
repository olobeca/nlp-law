#!/usr/bin/env python3
"""
Embed mock Kodeks Pracy articles with BGE-M3 and build a FAISS index.

Reads data/articles.json, generates L2-normalized embeddings, and saves:
  - index/kp_index.faiss
  - index/kp_metadata.json
"""

import json
import logging
import sys

from rag.config import ARTICLES_PATH, FAISS_INDEX_PATH, METADATA_PATH
from rag.embeddings import encode_documents, load_model
from rag.index_store import build_faiss_index, save_index, save_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_articles() -> list[dict]:
    """Load articles from the mock data file."""
    if not ARTICLES_PATH.exists():
        logger.error(
            "Articles file not found at %s. Add data/articles.json first.",
            ARTICLES_PATH,
        )
        sys.exit(1)

    with ARTICLES_PATH.open(encoding="utf-8") as f:
        articles = json.load(f)

    logger.info("Loaded %d articles from %s", len(articles), ARTICLES_PATH)
    return articles


def build_corpus(articles: list[dict]) -> list[str]:
    """Combine title and text for richer document embeddings."""
    return [f"{article['title']}\n{article['text']}" for article in articles]


def main() -> None:
    """Load articles, embed, index, and persist."""
    logger.info("=== Kodeks Pracy: embed and index ===")

    articles = load_articles()
    corpus = build_corpus(articles)
    logger.info("Built corpus of %d document strings", len(corpus))

    model = load_model()
    dim = model.get_sentence_embedding_dimension()
    logger.info("Embedding dimension: %d", dim)

    vectors = encode_documents(corpus)

    logger.info("Building FAISS IndexFlatIP index...")
    index = build_faiss_index(vectors)

    save_index(index, FAISS_INDEX_PATH)
    save_metadata(articles, METADATA_PATH)

    logger.info("=== Indexing complete ===")
    logger.info("Vectors indexed: %d", index.ntotal)
    logger.info("FAISS index: %s", FAISS_INDEX_PATH)
    logger.info("Metadata:    %s", METADATA_PATH)


if __name__ == "__main__":
    main()
