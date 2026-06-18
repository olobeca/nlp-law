"""Hybrid Search — łączy BM25 i Dense Retrieval przez Reciprocal Rank Fusion (RRF).

RRF scala dwa rankingi w jeden bez potrzeby normalizowania score'ów.
Wzór: score_rrf(doc) = 1/(rank_bm25 + K) + 1/(rank_dense + K)
gdzie K=60 to stała wygładzająca (standardowa wartość z literatury).

Przykład:
  BM25 zwraca:   [Art.36 (rank 1), Art.37 (rank 2), Art.52 (rank 3)]
  Dense zwraca:  [Art.52 (rank 1), Art.36 (rank 2), Art.42 (rank 3)]

  RRF score Art.36 = 1/(1+60) + 1/(2+60) = 0.0164 + 0.0161 = 0.0325
  RRF score Art.52 = 1/(3+60) + 1/(1+60) = 0.0159 + 0.0164 = 0.0323
  RRF score Art.37 = 1/(2+60) + 0        = 0.0161
  RRF score Art.42 = 0        + 1/(3+60) = 0.0159

  Wynik końcowy: [Art.36, Art.52, Art.37, Art.42]
"""

import logging

logger = logging.getLogger(__name__)

# Stała RRF — wartość 60 jest standardem z oryginalnej publikacji (Cormack 2009)
RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int,
) -> list[tuple[int, float]]:
    """
    Scala wyniki BM25 i Dense Retrieval przez RRF.

    Args:
        bm25_results:  Lista (idx_artykułu, score) z BM25, posortowana malejąco.
        dense_results: Lista (idx_artykułu, score) z FAISS, posortowana malejąco.
        k:             Liczba wyników końcowych.

    Returns:
        Lista (idx_artykułu, rrf_score) posortowana malejąco, max k elementów.
    """
    rrf_scores: dict[int, float] = {}

    # Dodaj wkład z rankingu BM25
    for rank, (doc_idx, _score) in enumerate(bm25_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rank + RRF_K)

    # Dodaj wkład z rankingu Dense
    for rank, (doc_idx, _score) in enumerate(dense_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rank + RRF_K)

    # Posortuj malejąco po RRF score i zwróć top-k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    logger.debug(
        "RRF: bm25=%d wyników, dense=%d wyników → hybrid=%d wyników",
        len(bm25_results),
        len(dense_results),
        len(ranked),
    )

    return ranked


def hybrid_search(
    bm25_model,
    faiss_index,
    query: str,
    query_vector,
    articles: list[dict],
    k: int,
) -> list[tuple[int, float]]:
    """
    Pełny pipeline Hybrid Search: BM25 + Dense → RRF.

    Args:
        bm25_model:   Indeks BM25 (z rag.bm25.build_bm25).
        faiss_index:  Indeks FAISS (z rag.index_store.load_index).
        query:        Pytanie użytkownika (tekst, dla BM25).
        query_vector: Embedding pytania (numpy array, dla FAISS).
        articles:     Lista wszystkich artykułów (metadata).
        k:            Liczba wyników końcowych.

    Returns:
        Lista (idx_artykułu, rrf_score) posortowana malejąco.
    """
    from rag.bm25 import search_bm25
    from rag.index_store import search as search_faiss

    # Pobierz więcej wyników niż k — RRF potrzebuje szerszego rankingu do scalenia
    fetch_k = min(k * 3, len(articles))

    # BM25 retrieval
    bm25_results = search_bm25(bm25_model, query, k=fetch_k)

    # Dense retrieval
    scores_arr, indices_arr = search_faiss(faiss_index, query_vector, k=fetch_k)
    dense_results = [
        (int(idx), float(score))
        for idx, score in zip(indices_arr[0], scores_arr[0])
        if idx >= 0
    ]

    # Scal przez RRF
    hybrid_results = reciprocal_rank_fusion(bm25_results, dense_results, k=k)

    return hybrid_results