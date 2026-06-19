import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    text = text.lower()

    text = re.sub(r'art\.?\s*(\d+\w*)', r'art_\1', text)
    text = re.sub(r'§\s*(\d+)', r'par_\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    tokens = text.split()

    tokens = [
        t for t in tokens
        if len(t) > 2 or t.startswith('art_') or t.startswith('par_')
    ]

    return tokens


def build_bm25(articles: list[dict]) -> BM25Okapi:
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
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = [
        (idx, float(scores[idx]))
        for idx in top_indices
        if scores[idx] > 0
    ]

    return results
