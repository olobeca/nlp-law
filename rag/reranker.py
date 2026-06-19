import logging
from typing import Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

from rag.config import RERANKER_MAX_CHARS, RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)

_reranker_cache: Dict[str, CrossEncoder] = {}


def load_reranker(model_name: Optional[str] = None) -> CrossEncoder:
    name = model_name or RERANKER_MODEL_NAME
    if name not in _reranker_cache:
        logger.info("Ładuję reranker: %s", name)
        _reranker_cache[name] = CrossEncoder(name)
        logger.info("Reranker załadowany: %s", name)
    return _reranker_cache[name]


def _article_text(article: dict) -> str:
    text = f"{article['title']}\n{article['text']}"
    if len(text) > RERANKER_MAX_CHARS:
        return text[:RERANKER_MAX_CHARS] + "..."
    return text


def rerank(
    query: str,
    results: List[Tuple[int, float]],
    metadata: List[dict],
    top_k: int,
    model_name: Optional[str] = None,
) -> List[Tuple[int, float]]:
    if not results:
        return []

    reranker = load_reranker(model_name)

    pairs: List[Tuple[str, str]] = []
    indices: List[int] = []
    for idx, _score in results:
        if idx < 0 or idx >= len(metadata):
            continue
        pairs.append((query, _article_text(metadata[idx])))
        indices.append(idx)

    if not pairs:
        return []

    logger.info("Rerankuję %d kandydatów → top %d", len(pairs), top_k)
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(indices, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [(int(idx), float(score)) for idx, score in ranked]
