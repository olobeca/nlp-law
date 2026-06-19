"""Ujednolicony pipeline retrieval z opcjonalnym rerankingiem."""

import logging
from typing import List, Tuple

from rag.bm25 import search_bm25
from rag.embeddings import encode_query
from rag.hybrid import hybrid_search
from rag.index_store import search as search_faiss
from rag.reranker import rerank

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    mode: str,
    top_k: int,
    index,
    bm25_model,
    metadata: List[dict],
    *,
    use_reranker: bool = False,
    rerank_candidates: int = 15,
    reranker_model: str = None,
) -> List[Tuple[int, float]]:
    """
    Wyszukaj artykuły wybranym trybem, opcjonalnie z rerankingiem cross-encoder.

    Gdy reranker włączony:
      1. Pobierz `rerank_candidates` dokumentów (szybki retrieval).
      2. Cross-encoder ocenia każdą parę (pytanie, artykuł).
      3. Zwróć `top_k` najlepszych po reranku.
    """
    fetch_k = rerank_candidates if use_reranker else top_k
    fetch_k = min(fetch_k, len(metadata))
    if fetch_k < 1:
        return []

    query_vector = encode_query(query)

    if mode == "dense":
        scores_arr, indices_arr = search_faiss(index, query_vector, fetch_k)
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices_arr[0], scores_arr[0])
            if idx >= 0
        ]
    elif mode == "bm25":
        results = search_bm25(bm25_model, query, k=fetch_k)
    else:
        results = hybrid_search(
            bm25_model, index, query, query_vector, metadata, k=fetch_k
        )

    if use_reranker and results:
        logger.info("Reranker włączony (%s), kandydaci: %d", mode, len(results))
        results = rerank(
            query,
            results,
            metadata,
            top_k=top_k,
            model_name=reranker_model,
        )
    else:
        results = results[:top_k]

    return results
