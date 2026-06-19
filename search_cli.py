#!/usr/bin/env python3
"""
Asystent prawny Kodeks Pracy — CLI z wyborem trybu wyszukiwania.

Tryby:
  1. hybrid  — BM25 + Dense scalony przez RRF (domyślny)
  2. dense   — tylko FAISS (embeddingi BGE-M3)
  3. bm25    — tylko BM25 (dopasowanie słów kluczowych)

Reranker (cross-encoder) domyślnie włączony — konfiguracja w rag/config.py.
"""

import logging
import sys

from rag.bm25 import build_bm25
from rag.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RERANKER_CANDIDATES_DEFAULT,
    RERANKER_ENABLED_DEFAULT,
    RERANKER_MODEL_NAME,
    TOP_K,
)
from rag.embeddings import load_model
from rag.index_store import load_index, load_metadata
from rag.llm import ask_llm
from rag.retrieve import retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def print_results(
    results: list[tuple[int, float]],
    metadata: list[dict],
    reranked: bool = False,
) -> None:
    score_label = "rerank" if reranked else "score"
    print()
    for rank, (idx, score) in enumerate(results, start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        article = metadata[idx]
        print(f"--- Wynik {rank} ({score_label}: {score:.4f}) ---")
        print(article["title"])
        print(article["text"][:300] + ("..." if len(article["text"]) > 300 else ""))
        print()


def choose_mode() -> str:
    print("\n╔══════════════════════════════════════╗")
    print("║   Kodeks Pracy — asystent prawny     ║")
    print("╠══════════════════════════════════════╣")
    print("║  Tryby wyszukiwania:                 ║")
    print("║  [1] hybrid — BM25 + Dense (domyślny)║")
    print("║  [2] dense  — tylko embeddingi       ║")
    print("║  [3] bm25   — tylko słowa kluczowe   ║")
    print("╚══════════════════════════════════════╝")

    choice = input("\nWybierz tryb [1/2/3] lub Enter dla domyślnego: ").strip()
    modes = {"1": "hybrid", "2": "dense", "3": "bm25", "": "hybrid"}
    mode = modes.get(choice, "hybrid")
    print(f"→ Tryb: {mode.upper()}\n")
    return mode


def run_search_loop(index, bm25_model, metadata, mode: str) -> None:
    use_reranker = RERANKER_ENABLED_DEFAULT
    rerank_label = "WŁĄCZONY" if use_reranker else "WYŁĄCZONY"
    print(f"Reranker: {rerank_label} (model: {RERANKER_MODEL_NAME})")
    print("Wpisz pytanie po polsku lub 'exit' aby zakończyć.")
    print("Tip: wpisz 'tryb' aby zmienić tryb wyszukiwania\n")

    while True:
        try:
            query = input("Pytanie: ").strip().strip('"\'')
        except (EOFError, KeyboardInterrupt):
            print("\nZakończono.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Do widzenia!")
            break

        if query.lower() == "tryb":
            mode = choose_mode()
            continue

        results = retrieve(
            query,
            mode,
            TOP_K,
            index,
            bm25_model,
            metadata,
            use_reranker=use_reranker,
            rerank_candidates=RERANKER_CANDIDATES_DEFAULT,
            reranker_model=RERANKER_MODEL_NAME,
        )

        mode_label = f"{mode.upper()}"
        if use_reranker:
            mode_label += " + RERANK"
        print(f"\n--- Znalezione artykuły [{mode_label}] ---")
        print_results(results, metadata, reranked=use_reranker)

        top_articles = [metadata[idx] for idx, _ in results if 0 <= idx < len(metadata)]
        answer = ask_llm(query, top_articles)

        print("=" * 50)
        print("ODPOWIEDŹ ASYSTENTA:")
        print("=" * 50)
        print(answer)
        print()


def main() -> None:
    logger.info("=== Kodeks Pracy: RAG asystent prawny ===")

    try:
        index = load_index(FAISS_INDEX_PATH)
        metadata = load_metadata(METADATA_PATH)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Buduję indeks BM25...")
    bm25_model = build_bm25(metadata)

    logger.info("Wczytuję model embeddingowy...")
    load_model()

    mode = choose_mode()
    run_search_loop(index, bm25_model, metadata, mode)


if __name__ == "__main__":
    main()
