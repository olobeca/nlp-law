import logging

logger = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int,
) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = {}

    for rank, (doc_idx, _score) in enumerate(bm25_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rank + RRF_K)

    for rank, (doc_idx, _score) in enumerate(dense_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rank + RRF_K)

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
    from rag.bm25 import search_bm25
    from rag.index_store import search as search_faiss

    fetch_k = min(k * 3, len(articles))

    bm25_results = search_bm25(bm25_model, query, k=fetch_k)

    scores_arr, indices_arr = search_faiss(faiss_index, query_vector, k=fetch_k)
    dense_results = [
        (int(idx), float(score))
        for idx, score in zip(indices_arr[0], scores_arr[0])
        if idx >= 0
    ]

    hybrid_results = reciprocal_rank_fusion(bm25_results, dense_results, k=k)

    return hybrid_results
