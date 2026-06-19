"""Streamlit chat UI — asystent RAG Kodeks Pracy."""

import os
import re
from typing import Dict, List, Optional, Tuple

import streamlit as st

from rag.bm25 import build_bm25
from rag.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RERANKER_CANDIDATES_DEFAULT,
    RERANKER_ENABLED_DEFAULT,
    RERANKER_MODELS,
    TOP_K,
)
from rag.embeddings import load_model
from rag.index_store import load_index, load_metadata
from rag.llm import MODEL, ask_llm
from rag.retrieve import retrieve

st.set_page_config(
    page_title="Kodeks Pracy — Asystent RAG",
    page_icon="⚖️",
    layout="wide",
)

SEARCH_MODES = {
    "hybrid": "Hybrid (BM25 + Dense)",
    "dense": "Dense (embeddingi)",
    "bm25": "BM25 (słowa kluczowe)",
}

ARTICLE_REF_PATTERN = re.compile(r"Art\.\s*(\d+)", re.IGNORECASE)


def _sidebar_separator() -> None:
    st.markdown("---")


def _article_sort_key(article: dict) -> int:
    match = re.search(r"(\d+)", article["title"])
    return int(match.group(1)) if match else article["id"]


def _build_article_indexes(metadata: List[dict]) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    by_id = {article["id"]: article for article in metadata}
    by_number: Dict[int, dict] = {}
    for article in metadata:
        match = re.search(r"(\d+)", article["title"])
        if match:
            by_number[int(match.group(1))] = article
    return by_id, by_number


def _find_article_by_number(article_number: int, by_number: Dict[int, dict]) -> Optional[dict]:
    return by_number.get(article_number)


def _extract_cited_articles(text: str, by_number: Dict[int, dict]) -> List[dict]:
    cited: List[dict] = []
    seen_ids: set = set()
    for match in ARTICLE_REF_PATTERN.finditer(text):
        article = _find_article_by_number(int(match.group(1)), by_number)
        if article and article["id"] not in seen_ids:
            cited.append(article)
            seen_ids.add(article["id"])
    return cited


def _open_article_in_library(article_id: int) -> None:
    st.session_state.page = "library"
    st.session_state.selected_article_id = article_id


@st.experimental_singleton(show_spinner=False)
def load_resources():
    with st.spinner("Ładuję indeksy i model embeddingowy..."):
        index = load_index(FAISS_INDEX_PATH)
        metadata = load_metadata(METADATA_PATH)
        bm25_model = build_bm25(metadata)
        load_model()
    return index, bm25_model, metadata


def render_sidebar_settings() -> tuple:
    st.header("Nawigacja")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("💬 Czat"):
            st.session_state.page = "chat"
            st.experimental_rerun()
    with nav_col2:
        if st.button("📚 Biblioteka"):
            st.session_state.page = "library"
            st.experimental_rerun()

    _sidebar_separator()
    st.header("Ustawienia")
    mode = st.selectbox(
        "Tryb wyszukiwania",
        options=list(SEARCH_MODES.keys()),
        format_func=lambda key: SEARCH_MODES[key],
        index=0,
    )
    top_k = st.slider("Liczba artykułów (TOP K)", min_value=1, max_value=10, value=TOP_K)
    _sidebar_separator()
    st.subheader("Reranker")
    use_reranker = st.checkbox(
        "Włącz cross-encoder reranker",
        value=RERANKER_ENABLED_DEFAULT,
        help="Drugi etap: precyzyjna ocena kandydatów z retrieval.",
    )
    rerank_candidates = st.slider(
        "Kandydaci przed rerankiem",
        min_value=top_k,
        max_value=30,
        value=max(RERANKER_CANDIDATES_DEFAULT, top_k),
        disabled=not use_reranker,
        help="Ile artykułów pobiera retrieval, zanim reranker wybierze TOP K.",
    )
    reranker_label = st.selectbox(
        "Model rerankera",
        options=list(RERANKER_MODELS.keys()),
        index=0,
        disabled=not use_reranker,
    )
    reranker_model = RERANKER_MODELS[reranker_label]
    _sidebar_separator()
    st.markdown(f"**Model LLM:** `{MODEL}`")
    st.markdown("**Embeddingi:** `BAAI/bge-m3`")
    if use_reranker:
        st.markdown(f"**Reranker:** `{reranker_model}`")
    if st.button("Wyczyść czat"):
        st.session_state.messages = []
        st.experimental_rerun()
    _sidebar_separator()
    st.caption(
        "To narzędzie informacyjne — nie zastępuje porady prawnej. "
        "Dane artykułów są mockami deweloperskimi."
    )
    return mode, top_k, use_reranker, rerank_candidates, reranker_model


def render_library_page(metadata: List[dict], by_id: Dict[int, dict]) -> None:
    st.title("📚 Biblioteka artykułów")
    st.caption(f"{len(metadata)} artykułów Kodeksu Pracy")

    search = st.text_input(
        "Szukaj",
        placeholder="np. urlop, Art. 154, wypowiedzenie...",
        key="library_search",
    ).strip().lower()

    selected_id = st.session_state.get("selected_article_id")
    if selected_id and selected_id in by_id:
        st.info(f"Wybrany artykuł: **{by_id[selected_id]['title']}**")
        if st.button("Odznacz artykuł"):
            st.session_state.selected_article_id = None
            st.experimental_rerun()

    sorted_articles = sorted(metadata, key=_article_sort_key)
    if search:
        articles = [
            a for a in sorted_articles
            if search in a["title"].lower() or search in a["text"].lower()
        ]
    else:
        articles = sorted_articles

    if not articles:
        st.warning("Brak artykułów pasujących do wyszukiwania.")
        return

    for article in articles:
        is_selected = article["id"] == selected_id
        label = f"{'▶ ' if is_selected else ''}{article['title']}"
        with st.expander(label, expanded=is_selected):
            st.markdown(article["text"])


def render_sources(
    results: List[tuple],
    metadata: List[dict],
    key_prefix: str,
    reranked: bool = False,
) -> None:
    if not results:
        st.warning("Nie znaleziono pasujących artykułów.")
        return

    with st.expander(f"Źródła ({len(results)} artykułów)", expanded=False):
        for rank, (idx, score) in enumerate(results, start=1):
            if idx < 0 or idx >= len(metadata):
                continue
            article = metadata[idx]
            col_text, col_btn = st.columns([5, 1])
            with col_text:
                score_label = "rerank" if reranked else "score"
                st.markdown(f"**{rank}. {article['title']}** — {score_label}: `{score:.4f}`")
                preview = article["text"]
                if len(preview) > 400:
                    preview = preview[:400] + "..."
                st.caption(preview)
            with col_btn:
                if st.button("→", key=f"{key_prefix}_src_{article['id']}_{rank}", help=f"Otwórz {article['title']}"):
                    _open_article_in_library(article["id"])
                    st.experimental_rerun()


def render_article_refs(articles: List[dict], key_prefix: str) -> None:
    if not articles:
        return

    st.markdown("**Odnośniki:**")
    cols = st.columns(min(len(articles), 5))
    for i, article in enumerate(articles):
        with cols[i % len(cols)]:
            if st.button(article["title"], key=f"{key_prefix}_ref_{article['id']}"):
                _open_article_in_library(article["id"])
                st.experimental_rerun()


def render_message(
    message: dict,
    metadata: List[dict],
    by_number: Dict[int, dict],
    message_idx: int,
) -> None:
    if message["role"] == "user":
        st.markdown(f"**Ty:** {message['content']}")
    else:
        st.markdown(f"**Asystent:** {message['content']}")

        cited = _extract_cited_articles(message["content"], by_number)
        if message.get("sources"):
            seen = {a["id"] for a in cited}
            for idx, _ in message["sources"]:
                if 0 <= idx < len(metadata) and metadata[idx]["id"] not in seen:
                    cited.append(metadata[idx])
                    seen.add(metadata[idx]["id"])

        render_article_refs(cited, key_prefix=f"msg_{message_idx}")
        if message.get("sources"):
            render_sources(
                message["sources"],
                metadata,
                key_prefix=f"msg_{message_idx}",
                reranked=message.get("reranked", False),
            )

    st.markdown("---")


def render_chat_page(
    metadata: List[dict],
    by_number: Dict[int, dict],
    mode: str,
    top_k: int,
    use_reranker: bool,
    rerank_candidates: int,
    reranker_model: str,
    index,
    bm25_model,
) -> None:
    st.subheader("Czat")
    for i, message in enumerate(st.session_state.messages):
        render_message(message, metadata, by_number, message_idx=i)

    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Zadaj pytanie o Kodeks Pracy...")
        submitted = st.form_submit_button("Wyślij")

    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Szukam artykułów i generuję odpowiedź..."):
            results = retrieve(
                prompt,
                mode,
                top_k,
                index,
                bm25_model,
                metadata,
                use_reranker=use_reranker,
                rerank_candidates=rerank_candidates,
                reranker_model=reranker_model,
            )
            top_articles = [
                metadata[idx] for idx, _ in results if 0 <= idx < len(metadata)
            ]

            if not top_articles:
                answer = "Nie znalazłem pasujących artykułów w bazie."
            else:
                answer = ask_llm(prompt, top_articles)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": results,
                "reranked": use_reranker,
            }
        )
        st.experimental_rerun()


def main() -> None:
    st.title("⚖️ Kodeks Pracy — Asystent RAG")
    st.caption(
        "Zadaj pytanie po polsku. System wyszuka artykuły i wygeneruje odpowiedź na ich podstawie."
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("Brak klucza `GROQ_API_KEY`. Ustaw go w pliku `.env` w katalogu projektu.")
        st.stop()

    try:
        index, bm25_model, metadata = load_resources()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Uruchom najpierw: `python embed_and_index.py`")
        st.stop()

    by_id, by_number = _build_article_indexes(metadata)

    if "page" not in st.session_state:
        st.session_state.page = "chat"
    if "selected_article_id" not in st.session_state:
        st.session_state.selected_article_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        mode, top_k, use_reranker, rerank_candidates, reranker_model = render_sidebar_settings()

    if st.session_state.page == "library":
        render_library_page(metadata, by_id)
    else:
        render_chat_page(
            metadata,
            by_number,
            mode,
            top_k,
            use_reranker,
            rerank_candidates,
            reranker_model,
            index,
            bm25_model,
        )


if __name__ == "__main__":
    main()
