"""BM25 retriever dla artykułów Kodeksu Pracy.

BM25 wyszukuje po dokładnych słowach — uzupełnia Dense Retrieval
który wyszukuje po znaczeniu. Szczególnie pomocny przy:
- numerach artykułów: "art. 36", "art. 52"
- terminach prawnych: "wypowiedzenie", "dyscyplinarka"
- pytaniach z konkretnymi słowami kluczowymi
"""

import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    Prosta tokenizacja dla BM25.

    - zamiana na małe litery
    - zachowanie numerów artykułów jako jednego tokenu (art.36 → art_36)
    - usunięcie interpunkcji
    - podział na słowa
    """
    text = text.lower()

    # Zachowaj numery artykułów jako jeden token — wszystkie warianty:
    # "art. 36"  "art.36"  "art 36"  "art36"  → art_36
    text = re.sub(r'art\.?\s*(\d+\w*)', r'art_\1', text)

    # Zachowaj paragrafy jako token: "§ 1" "§1" → par_1
    text = re.sub(r'§\s*(\d+)', r'par_\1', text)

    # Usuń interpunkcję (zostaw litery, cyfry, podkreślenia)
    text = re.sub(r'[^\w\s]', ' ', text)

    tokens = text.split()

    # Usuń bardzo krótkie tokeny (1-2 znaki) oprócz art_XX i par_XX
    tokens = [
        t for t in tokens
        if len(t) > 2 or t.startswith('art_') or t.startswith('par_')
    ]

    return tokens


def build_bm25(articles: list[dict]) -> BM25Okapi:
    """
    Buduje indeks BM25 z listy artykułów.

    Wywołaj raz przy starcie programu — indeks trzymaj w pamięci.

    Args:
        articles: Lista słowników z kluczami 'title' i 'text'.

    Returns:
        Gotowy indeks BM25Okapi.
    """
    corpus = [
        _tokenize(f"{a['title']} {a['text']}")
        for a in articles
    ]
    bm25 = BM25Okapi(corpus)
    logger.info("BM25 index built: %d documents", len(corpus))
    return bm25


def search_bm25(
    bm25: BM25Okapi,
    query: str,
    k: int,
) -> list[tuple[int, float]]:
    """
    Wyszukuje top-k artykułów przez BM25.

    Args:
        bm25:  Indeks zbudowany przez build_bm25().
        query: Pytanie użytkownika.
        k:     Liczba wyników do zwrócenia.

    Returns:
        Lista (index_artykułu, score) posortowana malejąco po score.
        Zwraca tylko artykuły z score > 0.
    """
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    # Pobierz top-k indeksów posortowanych malejąco
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = [
        (idx, float(scores[idx]))
        for idx in top_indices
        if scores[idx] > 0  # pomiń artykuły bez żadnego dopasowania
    ]

    return results