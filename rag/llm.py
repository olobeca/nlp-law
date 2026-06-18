"""Generowanie odpowiedzi przez LLM (Groq) na podstawie artykułów z FAISS."""

import logging
import os

import rag.config  # noqa: F401 — ładuje .env
from groq import Groq

logger = logging.getLogger(__name__)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """Jesteś asystentem prawnym specjalizującym się w polskim prawie pracy.

Otrzymujesz listę artykułów Kodeksu Pracy oraz pytanie użytkownika.
Twoim zadaniem jest udzielenie odpowiedzi na podstawie tych artykułów.

Zasady:
1. Przeczytaj WSZYSTKIE dostarczone artykuły zanim odpiszesz.
2. Jeśli którykolwiek artykuł zawiera informację odpowiadającą na pytanie — użyj jej.
3. Przy każdej informacji podaj źródło, np. (Art. 154 KP).
4. Odpowiadaj po polsku, zwięźle i konkretnie.
5. Napisz "Nie znalazłem odpowiedzi w dostarczonych artykułach" TYLKO gdy żaden
   z artykułów nie zawiera żadnej istotnej informacji na temat pytania."""


def ask_llm(query: str, articles: list[dict]) -> str:
    """
    Wysyła pytanie i pasujące artykuły do LLM, zwraca gotową odpowiedź.

    Args:
        query:    Pytanie użytkownika.
        articles: Lista słowników z kluczami 'title' i 'text'.

    Returns:
        Odpowiedź LLM jako string.
    """
    user_message = _build_user_message(query, articles)

    logger.info("Wysyłam zapytanie do LLM (model: %s)...", MODEL)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    logger.info("Odpowiedź LLM otrzymana (%d znaków).", len(answer))
    return answer


def _build_user_message(query: str, articles: list[dict]) -> str:
    context = "\n\n".join(
        f"[{article['title']}]\n{article['text']}"
        for article in articles
    )
    return (
        f"Poniżej znajdują się artykuły Kodeksu Pracy dobrane do pytania.\n"
        f"Przeczytaj je uważnie i odpowiedz na pytanie.\n\n"
        f"ARTYKUŁY:\n{context}\n\n"
        f"PYTANIE: {query}"
    )


def ask_llm_stream(query: str, articles: list[dict]):
    """Strumieniuje odpowiedź LLM token po tokenie (generator dla Streamlit)."""
    user_message = _build_user_message(query, articles)

    logger.info("Strumieniuję odpowiedź LLM (model: %s)...", MODEL)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta