"""
Parser Kodeksu Pracy
====================
Pobiera jednolity tekst KP z api.sejm.gov.pl i parsuje go na artykuły.

Użycie:
    pip install pdfplumber requests
    python parser_kodeks_pracy.py

Wynik:
    kodeks_pracy_artykuly.json
"""

import re
import json
import requests
import pdfplumber
from pathlib import Path

# ── Konfiguracja ──────────────────────────────────────────────────────────────

PDF_URL  = "https://api.sejm.gov.pl/eli/acts/DU/2025/277/text.pdf"
PDF_PATH = "kodeks_pracy.pdf"
OUT_PATH = "kodeks_pracy_artykuly.json"

# Marker który jednoznacznie wskazuje początek właściwego tekstu KP.
# Wszystko przed nim to preambuła obwieszczenia — pomijamy.
KP_START_MARKER = "USTAWA\nz dnia 26 czerwca 1974 r.\nKodeks pracy"
KP_START_MARKER_ALT = "USTAWA z dnia 26 czerwca 1974 r. Kodeks pracy"


# ── 1. Pobieranie PDF ─────────────────────────────────────────────────────────

def download_pdf(url: str, path: str) -> None:
    """Pobiera PDF jeśli jeszcze nie istnieje na dysku."""
    if Path(path).exists():
        print(f"[INFO] PDF już istnieje: {path} — pomijam pobieranie.")
        return

    print(f"[INFO] Pobieram PDF z {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    print(f"[INFO] Zapisano: {path} ({len(response.content) / 1024:.0f} KB)")


# ── 2. Wyciąganie tekstu z PDF ────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Wyciąga pełny tekst z PDF strona po stronie."""
    print(f"[INFO] Czytam PDF: {pdf_path} ...")
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                pages.append(text)
            if i % 20 == 0:
                print(f"  strona {i}/{total}")

    full_text = "\n".join(pages)
    print(f"[INFO] Wyciągnięto {len(full_text):,} znaków z {total} stron.")
    return full_text


# ── 3. Czyszczenie i przycięcie preambuły ────────────────────────────────────

def clean_text(text: str) -> str:
    """
    1. Przycina tekst do początku właściwego Kodeksu pracy
       (usuwa preambuły obwieszczenia z listami dyrektyw EU).
    2. Usuwa artefakty PDF: nagłówki/stopki, numery stron,
       wyrazy dzielone myślnikiem, nadmiarowe spacje.
    """
    # Znajdź początek właściwego tekstu KP
    cut_pos = text.find(KP_START_MARKER)
    if cut_pos == -1:
        cut_pos = text.find(KP_START_MARKER_ALT)
    if cut_pos == -1:
        # Fallback: szukaj pierwszego "Art. 1." które NIE jest w cytatach
        cut_pos = 0
        print("[WARN] Nie znaleziono markera początku KP — przetwarzam cały tekst.")
    else:
        print(f"[INFO] Preambuła pominięta — KP zaczyna się na pozycji {cut_pos:,}.")
        text = text[cut_pos:]

    # Nagłówki i stopki stron
    text = re.sub(r'Dziennik Ustaw[^\n]*\n', '', text)
    text = re.sub(r'–\s*\d+\s*–', '', text)

    # Wyrazy dzielone myślnikiem na końcu linii (wy-\npowiedzenie → wypowiedzenie)
    text = re.sub(r'-\n\s*', '', text)

    # Wielokrotne spacje i puste linie
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── 4. Parsowanie artykułów ───────────────────────────────────────────────────

def parse_articles(text: str) -> list[dict]:
    """
    Dzieli tekst na artykuły.

    Obsługuje formaty numerów:
      Art. 1.       — zwykły artykuł
      Art. 29(1).   — artykuł z indeksem cyfrowym w nawiasie
      Art. 183a.    — artykuł z literą (183a, 183b, 183c, 183d, 183e...)
      Art. 151(1).  — artykuły trzycyfrowe z indeksem

    Każdy artykuł to słownik:
      {
        "id":    1,
        "title": "Art. 1",
        "text":  "pełny tekst artykułu..."
      }
    """

    # Wzorzec: Art. + numer (cyfry + opcjonalna litera lub (cyfra)) + kropka
    # Musi stać na początku linii lub po białych znakach
    # Obsługuje: Art. 1.  Art. 29(1).  Art. 183a.  Art. 151(1).
    ARTICLE_START = re.compile(
        r'(?m)^Art\.\s*(\d+(?:[a-z]|\(\d+\))?)\.'
    )

    matches = list(ARTICLE_START.finditer(text))
    print(f"[INFO] Znaleziono {len(matches)} artykułów.")

    # Usuń duplikaty — zachowaj tylko pierwsze wystąpienie każdego numeru artykułu.
    # Duplikaty powstają gdy ten sam artykuł jest cytowany w przypisach lub
    # obwieszczeniu nowelizacyjnym.
    seen_nums  = set()
    deduped    = []
    duplicates = 0
    for match in matches:
        num = match.group(1)
        if num not in seen_nums:
            seen_nums.add(num)
            deduped.append(match)
        else:
            duplicates += 1

    if duplicates:
        print(f"[INFO] Pominięto {duplicates} zduplikowanych artykułów.")

    articles = []
    for i, match in enumerate(deduped):
        num_raw = match.group(1)
        start   = match.start()
        end     = deduped[i + 1].start() if i + 1 < len(deduped) else len(text)

        article_text = text[start:end].strip()
        article_text = _clean_article_text(article_text)

        if len(article_text) < 20:
            continue

        articles.append({
            "id":    len(articles) + 1,   # ciągłe ID bez przerw
            "title": f"Art. {num_raw}",
            "text":  article_text,
        })

    return articles


def _clean_article_text(text: str) -> str:
    """Dodatkowe czyszczenie pojedynczego artykułu."""
    # Usuń nagłówki sekcji które wpadły w środek artykułu
    text = re.sub(r'\n(Dział|Rozdział|Oddział)\s+[IVXLC\d]+[^\n]*\n', '\n', text)

    # Usuń przypisy dolne — linie zaczynające się od cyfry i nawiasu: 1) lub 1.
    text = re.sub(r'\n\d+\)\s[A-ZŁŚŻŹ]', '', text)

    # Normalizuj paragrafy
    text = re.sub(r'§\s+', '§ ', text)

    # Spłaszcz newliny do spacji (jeden długi string — lepszy do embeddingu)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# ── 5. Zapis do JSON ──────────────────────────────────────────────────────────

def save_json(articles: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Zapisano {len(articles)} artykułów → {path}")


# ── 6. Podgląd wyniku ─────────────────────────────────────────────────────────

def preview(articles: list[dict], n: int = 3) -> None:
    print(f"\n{'='*60}")
    print(f"PODGLĄD — pierwsze {n} artykuły:")
    print('='*60)
    for art in articles[:n]:
        print(f"\nID:    {art['id']}")
        print(f"Tytuł: {art['title']}")
        print(f"Tekst: {art['text'][:200]}...")
    print('='*60)
    print(f"\nOstatni artykuł: {articles[-1]['title']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Krok 1 — pobierz PDF
    download_pdf(PDF_URL, PDF_PATH)

    # Krok 2 — wyciągnij tekst
    raw_text = extract_text(PDF_PATH)

    # Krok 3 — oczyść i utnij preambuły
    clean = clean_text(raw_text)

    # Krok 4 — parsuj artykuły
    articles = parse_articles(clean)

    # Krok 5 — zapisz JSON
    save_json(articles, OUT_PATH)

    # Krok 6 — podgląd
    preview(articles)

    print(f"\n✓ Gotowe! Wynik w pliku: {OUT_PATH}")


if __name__ == "__main__":
    main()