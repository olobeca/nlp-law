#!/usr/bin/env python3
"""
Asystent prawny Kodeks Pracy — CLI z wyborem trybu wyszukiwania.

Tryby:
  1. dense  — tylko FAISS (embeddingi BGE-M3)
  2. bm25   — tylko BM25 (dopasowanie słów kluczowych)
  3. hybrid — BM25 + Dense scalony przez RRF (domyślny, najlepszy)
"""

import logging
import sys

from rag.bm25 import build_bm25, search_bm25
from rag.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K
from rag.embeddings import encode_query, load_model
from rag.hybrid import hybrid_search
from rag.index_store import load_index, load_metadata, search as search_faiss
from rag.llm import ask_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Wyświetlanie wyników ──────────────────────────────────────────────────────

def print_results(results: list[tuple[int, float]], metadata: list[dict]) -> None:
    """Wyświetl artykuły z ich score'ami."""
    print()
    for rank, (idx, score) in enumerate(results, start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        article = metadata[idx]
        print(f"--- Wynik {rank} (score: {score:.4f}) ---")
        print(article["title"])
        print(article["text"][:300] + ("..." if len(article["text"]) > 300 else ""))
        print()


# ── Wybór trybu ───────────────────────────────────────────────────────────────

def choose_mode() -> str:
    """Zapytaj użytkownika o tryb wyszukiwania."""
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


# ── Główna pętla ──────────────────────────────────────────────────────────────

def run_search_loop(index, bm25_model, metadata, mode: str) -> None:
    """Interaktywna pętla pytań."""
    print(f"Wpisz pytanie po polsku lub 'exit' aby zakończyć.")
    print(f"Tip: możesz zmienić tryb wpisując 'tryb'\n")

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

        # Zawsze generuj embedding (potrzebny dla dense i hybrid)
        query_vector = encode_query(query)

        # ── Retrieval ─────────────────────────────────────────────────────────
        if mode == "dense":
            scores_arr, indices_arr = search_faiss(index, query_vector, TOP_K)
            results = [
                (int(idx), float(score))
                for idx, score in zip(indices_arr[0], scores_arr[0])
                if idx >= 0
            ]

        elif mode == "bm25":
            results = search_bm25(bm25_model, query, k=TOP_K)

        else:  # hybrid
            results = hybrid_search(
                bm25_model, index, query, query_vector, metadata, k=TOP_K
            )

        # ── Wyświetl artykuły ─────────────────────────────────────────────────
        print(f"\n--- Znalezione artykuły [{mode.upper()}] ---")
        print_results(results, metadata)

        # ── Generowanie odpowiedzi przez LLM ──────────────────────────────────
        top_articles = [metadata[idx] for idx, _ in results if 0 <= idx < len(metadata)]
        answer = ask_llm(query, top_articles)

        print("=" * 50)
        print("ODPOWIEDŹ ASYSTENTA:")
        print("=" * 50)
        print(answer)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Kodeks Pracy: RAG asystent prawny ===")

    # Wczytaj FAISS
    try:
        index = load_index(FAISS_INDEX_PATH)
        metadata = load_metadata(METADATA_PATH)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Zbuduj BM25 z tych samych artykułów
    logger.info("Buduję indeks BM25...")
    bm25_model = build_bm25(metadata)

    # Wczytaj model embeddingowy
    logger.info("Wczytuję model embeddingowy...")
    load_model()

    # Wybierz tryb i uruchom pętlę
    mode = choose_mode()
    run_search_loop(index, bm25_model, metadata, mode)


if __name__ == "__main__":
    main()