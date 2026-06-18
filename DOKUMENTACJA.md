# Dokumentacja projektu — Kodeks Pracy RAG

Przewodnik krok po kroku po wszystkich plikach w projekcie. Opisuje, co robi każdy element, jak się łączy z resztą systemu i pokazuje kluczowe fragmenty kodu.

---

## Jak to działa — ogólny przepływ

```
data/articles.json          ← dataset (źródło prawdy)
        │
        ▼
embed_and_index.py          ← koduje artykuły → buduje indeks FAISS
        │
        ├── index/kp_index.faiss      (wektory liczbowe)
        └── index/kp_metadata.json    (teksty powiązane z wektorami)
                │
                ▼
        search_cli.py         ← przyjmuje zapytanie po polsku → zwraca TOP 5 artykułów
```

Wspólną logikę (model, FAISS, ścieżki) trzyma pakiet `rag/`.

---

## Krok 1 — `requirements.txt`

Lista bibliotek Pythona wymaganych do działania projektu.

```txt
sentence-transformers>=3.0.0   # ładuje model BGE-M3 i generuje embeddingi
faiss-cpu>=1.8.0               # wyszukiwanie wektorowe (indeks FAISS)
numpy>=1.26.0                  # operacje na macierzach wektorów
torch>=2.0.0                   # backend neuronowy dla sentence-transformers
tqdm>=4.66.0                   # pasek postępu przy kodowaniu
```

Instalacja: `pip install -r requirements.txt` (wewnątrz aktywnego venv).

---

## Krok 2 — `data/articles.json`

**Rola:** Dataset — baza artykułów Kodeksu Pracy. To jedyne źródło tekstów w systemie.

**Format:** tablica obiektów JSON. Każdy artykuł ma trzy pola:

```json
{
  "id": 16,
  "title": "Art. 154",
  "text": "Pracownikowi przysługuje coroczny urlop wypoczynkowy w wymiarze 20 dni roboczych..."
}
```

| Pole    | Typ    | Opis                                      |
|---------|--------|-------------------------------------------|
| `id`    | int    | Unikalny identyfikator artykułu (1–30)    |
| `title` | string | Numer artykułu, np. `"Art. 154"`          |
| `text`  | string | Treść artykułu po polsku                  |

Plik jest commitowany do repozytorium. Gdy zmienisz artykuły, musisz ponownie uruchomić `embed_and_index.py`, żeby przebudować indeks.

---

## Krok 3 — `rag/config.py`

**Rola:** Centralna konfiguracja — jedno miejsce na wszystkie stałe i ścieżki plików.

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "BAAI/bge-m3"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ARTICLES_PATH = PROJECT_ROOT / "data" / "articles.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "index" / "kp_index.faiss"
METADATA_PATH = PROJECT_ROOT / "index" / "kp_metadata.json"

TOP_K = 5
```

**Co tu jest ważne:**

- `MODEL_NAME` — model embeddingowy z Hugging Face. Wielojęzyczny, dobry do polskiego.
- `QUERY_PREFIX` — specjalny prefiks dodawany tylko do zapytań użytkownika (asymetryczne wyszukiwanie BGE).
- Ścieżki są budowane względem katalogu projektu, więc skrypty działają niezależnie od tego, skąd je uruchomisz.
- `TOP_K = 5` — ile wyników zwraca wyszukiwarka.

---

## Krok 4 — `rag/embeddings.py`

**Rola:** Ładowanie modelu BGE-M3 i zamiana tekstu na wektory liczbowe (embeddingi).

### 4a. Ładowanie modelu (singleton)

Model ładuje się **raz** i jest współdzielony między wywołaniami:

```python
_model: Optional[SentenceTransformer] = None

def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model
```

Dzięki temu `search_cli.py` nie przeładowuje ~2 GB modelu przy każdym zapytaniu.

### 4b. Kodowanie dokumentów (przy indeksowaniu)

```python
def encode_documents(texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # L2-normalizacja → cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)
```

Każdy artykuł zamienia się w wektor 1024 liczb. `normalize_embeddings=True` normalizuje wektory do długości 1 — to kluczowe dla cosine similarity przez FAISS.

### 4c. Kodowanie zapytania (przy wyszukiwaniu)

```python
def encode_query(query: str) -> np.ndarray:
    prefixed = QUERY_PREFIX + query
    vector = model.encode([prefixed], normalize_embeddings=True, ...)
    return vector.astype(np.float32)
```

Zapytania dostają prefiks instrukcji BGE (`"Represent this sentence for searching relevant passages: "`), dokumenty — nie. To poprawia trafność wyszukiwania.

**Przykład:** zapytanie `"Ile dni urlopu mi przysługuje?"` jest kodowane jako:
```
"Represent this sentence for searching relevant passages: Ile dni urlopu mi przysługuje?"
```

---

## Krok 5 — `rag/index_store.py`

**Rola:** Budowanie, zapis, odczyt i przeszukiwanie indeksu FAISS oraz metadanych.

### 5a. Budowanie indeksu

```python
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]          # 1024 dla BGE-M3
    index = faiss.IndexFlatIP(dim)  # Inner Product
    index.add(vectors)
    return index
```

`IndexFlatIP` liczy iloczyn skalarny (inner product). Na znormalizowanych wektorach iloczyn skalarny = cosine similarity. Score bliższy `1.0` = większe podobieństwo.

### 5b. Zapis i odczyt

```python
faiss.write_index(index, str(path))   # zapis → kp_index.faiss
faiss.read_index(str(path))          # odczyt przy wyszukiwaniu
```

Metadane zapisywane są jako JSON — **kolejność elementów w tablicy musi odpowiadać numerom wierszy w FAISS**:

```python
# Wiersz 0 w FAISS → metadata[0] → Art. 22
# Wiersz 15 w FAISS → metadata[15] → Art. 154
```

### 5c. Wyszukiwanie

```python
def search(index, query_vector, k) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query_vector, k)
    return scores, indices
```

Zwraca dwie tablice: `scores` (podobieństwo) i `indices` (numery wierszy w indeksie). Te indeksy służą do pobrania tekstu z `kp_metadata.json`.

---

## Krok 6 — `embed_and_index.py`

**Rola:** Główny skrypt indeksowania. Uruchamiasz go **raz** (lub po każdej zmianie datasetu).

**Kolejność kroków w `main()`:**

```
1. load_articles()     → wczytaj data/articles.json
2. build_corpus()      → połącz tytuł + tekst każdego artykułu
3. load_model()        → załaduj BGE-M3 (pierwszy raz: pobiera ~2 GB)
4. encode_documents()  → zamień 30 tekstów na 30 wektorów
5. build_faiss_index() → zbuduj indeks FAISS
6. save_index()        → zapisz index/kp_index.faiss
7. save_metadata()     → zapisz index/kp_metadata.json
```

### Kluczowy fragment — budowanie korpusu

```python
def build_corpus(articles: list[dict]) -> list[str]:
    return [f"{article['title']}\n{article['text']}" for article in articles]
```

Tytuł i tekst są łączone, żeby embedding zawierał też numer artykułu. Dzięki temu zapytanie typu „urlop" trafia na Art. 154, a nie na przypadkowy artykuł.

### Uruchomienie

```bash
python embed_and_index.py
```

Typowy output:
```
[INFO] Loaded 30 articles from .../data/articles.json
[INFO] Embedding dimension: 1024
[INFO] FAISS index built: 30 vectors, dim=1024
[INFO] === Indexing complete ===
```

---

## Krok 7 — `search_cli.py`

**Rola:** Interaktywna wyszukiwarka w terminalu. Ładuje gotowy indeks i odpowiada na zapytania po polsku.

### 7a. Start — ładowanie zasobów

```python
index = load_index(FAISS_INDEX_PATH)       # wektory z dysku
metadata = load_metadata(METADATA_PATH)    # teksty z dysku
load_model()                               # model BGE-M3 (z cache)
```

### 7b. Pętla interaktywna

```python
while True:
    query = input("Zapytanie (lub 'exit'): ").strip()
    if query.lower() in ("exit", "quit", "q"):
        break

    query_vector = encode_query(query)              # tekst → wektor
    scores, indices = search(index, query_vector, TOP_K)  # szukaj TOP 5
    print_results(scores, indices, metadata)        # wyświetl wyniki
```

### 7c. Wyświetlanie wyników

```python
def print_results(scores, indices, metadata):
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        article = metadata[idx]   # idx → numer wiersza FAISS → tekst artykułu
        print(f"--- Wynik {rank} (score: {score:.4f}) ---")
        print(article["title"])
        print(article["text"])
```

### Przykładowa sesja

```
Zapytanie (lub 'exit'): Ile dni urlopu mi przysługuje?

--- Wynik 1 (score: 0.6151) ---
Art. 154
Pracownikowi przysługuje coroczny urlop wypoczynkowy w wymiarze 20 dni roboczych...

--- Wynik 2 (score: 0.6021) ---
Art. 162
Pracownikowi przysługuje urlop na żądanie w wymiarze 4 dni w roku kalendarzowym...
```

### Uruchomienie

```bash
python search_cli.py
```

---

## Pliki generowane (nie edytuj ręcznie)

| Plik | Tworzony przez | Zawartość |
|------|---------------|-----------|
| `index/kp_index.faiss` | `embed_and_index.py` | Binarny indeks FAISS — 30 wektorów × 1024 wymiary |
| `index/kp_metadata.json` | `embed_and_index.py` | Kopia artykułów w tej samej kolejności co wektory |

Te pliki są w `.gitignore` — generujesz je lokalnie po `pip install`.

---

## Diagram zależności między plikami

```
requirements.txt
       │
       ▼
rag/config.py ◄────────────────────────────────────┐
       │                                          │
       ├──► rag/embeddings.py                     │
       │         ▲                                │
       │         │                                │
       └──► rag/index_store.py                   │
                 ▲                                │
                 │                                │
data/articles.json                                │
       │                                          │
       ▼                                          │
embed_and_index.py ──────────────────────────────┘
       │
       ├──► index/kp_index.faiss
       └──► index/kp_metadata.json
                │
                ▼
         search_cli.py
```

---

## Typowy workflow

```bash
# Jednorazowo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Po każdej zmianie articles.json
python embed_and_index.py

# Wyszukiwanie (wielokrotnie)
python search_cli.py
```

**Pierwsze uruchomienie `embed_and_index.py`** trwa kilka minut — pobiera model BGE-M3 (~2 GB). Kolejne uruchomienia są szybkie (model jest w cache Hugging Face).
