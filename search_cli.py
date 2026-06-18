#!/usr/bin/env python3
"""
Interactive CLI for semantic search over Kodeks Pracy articles.

Loads the FAISS index and metadata, then runs a REPL for Polish queries.
"""

import logging
import sys

from rag.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from rag.embeddings import encode_query, load_model
from rag.index_store import load_index, load_metadata, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def print_results(
    scores: list[float],
    indices: list[int],
    metadata: list[dict],
) -> None:
    """Print ranked search results with titles, texts, and similarity scores."""
    print()
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        if idx < 0:
            continue
        article = metadata[idx]
        print(f"--- Wynik {rank} (score: {score:.4f}) ---")
        print(article["title"])
        print(article["text"])
        print()


def run_search_loop(index, metadata) -> None:
    """Interactive query loop."""
    print("\nKodeks Pracy — wyszukiwanie semantyczne")
    print(f"Zwracane wyniki: TOP {TOP_K}")
    print("Wpisz zapytanie po polsku lub 'exit' aby zakończyć.\n")

    while True:
        try:
            query = input("Zapytanie (lub 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nZakończono.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Do widzenia!")
            break

        query_vector = encode_query(query)
        scores_arr, indices_arr = search(index, query_vector, TOP_K)

        scores = scores_arr[0].tolist()
        indices = indices_arr[0].tolist()
        print_results(scores, indices, metadata)


def main() -> None:
    """Load index, metadata, and model; start interactive search."""
    logger.info("=== Kodeks Pracy: search CLI ===")

    try:
        index = load_index(FAISS_INDEX_PATH)
        metadata = load_metadata(METADATA_PATH)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Loading embedding model for queries...")
    load_model()

    run_search_loop(index, metadata)


if __name__ == "__main__":
    main()
