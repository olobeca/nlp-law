import re
import json
import requests
import pdfplumber
from pathlib import Path

PDF_URL = "https://api.sejm.gov.pl/eli/acts/DU/2025/277/text.pdf"
PDF_PATH = "kodeks_pracy.pdf"
OUT_PATH = "articles.json"

KP_START_MARKER = "USTAWA\nz dnia 26 czerwca 1974 r.\nKodeks pracy"
KP_START_MARKER_ALT = "USTAWA z dnia 26 czerwca 1974 r. Kodeks pracy"


def download_pdf(url: str, path: str) -> None:
    if Path(path).exists():
        print(f"[INFO] PDF już istnieje: {path} — pomijam pobieranie.")
        return

    print(f"[INFO] Pobieram PDF z {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(path, "wb") as f:
        f.write(response.content)

    print(f"[INFO] Zapisano: {path} ({len(response.content) / 1024:.0f} KB)")


def extract_text(pdf_path: str) -> str:
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


def clean_text(text: str) -> str:
    cut_pos = text.find(KP_START_MARKER)
    if cut_pos == -1:
        cut_pos = text.find(KP_START_MARKER_ALT)
    if cut_pos == -1:
        cut_pos = 0
        print("[WARN] Nie znaleziono markera początku KP — przetwarzam cały tekst.")
    else:
        print(f"[INFO] Preambuła pominięta — KP zaczyna się na pozycji {cut_pos:,}.")
        text = text[cut_pos:]

    text = re.sub(r'Dziennik Ustaw[^\n]*\n', '', text)
    text = re.sub(r'–\s*\d+\s*–', '', text)
    text = re.sub(r'-\n\s*', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def parse_articles(text: str) -> list[dict]:
    ARTICLE_START = re.compile(
        r'(?m)^Art\.\s*(\d+(?:[a-z]|\(\d+\))?)\.'
    )

    matches = list(ARTICLE_START.finditer(text))
    print(f"[INFO] Znaleziono {len(matches)} artykułów.")

    seen_nums = set()
    deduped = []
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
        start = match.start()
        end = deduped[i + 1].start() if i + 1 < len(deduped) else len(text)

        article_text = text[start:end].strip()
        article_text = _clean_article_text(article_text)

        if len(article_text) < 20:
            continue

        articles.append({
            "id": len(articles) + 1,
            "title": f"Art. {num_raw}",
            "text": article_text,
        })

    return articles


def _clean_article_text(text: str) -> str:
    text = re.sub(r'\n(Dział|Rozdział|Oddział)\s+[IVXLC\d]+[^\n]*\n', '\n', text)
    text = re.sub(r'\n\d+\)\s[A-ZŁŚŻŹ]', '', text)
    text = re.sub(r'§\s+', '§ ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def save_json(articles: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Zapisano {len(articles)} artykułów → {path}")


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


def main():
    download_pdf(PDF_URL, PDF_PATH)
    raw_text = extract_text(PDF_PATH)
    clean = clean_text(raw_text)
    articles = parse_articles(clean)
    save_json(articles, OUT_PATH)
    preview(articles)
    print(f"\n✓ Gotowe! Wynik w pliku: {OUT_PATH}")


if __name__ == "__main__":
    main()
