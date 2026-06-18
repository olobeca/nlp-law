# nlp-law — Kodeks Pracy Local RAG

Local Retrieval-Augmented Generation (RAG) subsystem for Polish Labor Code research. Uses **BAAI/bge-m3** embeddings and **FAISS** for semantic search over Kodeks Pracy articles.

> **Note:** Article texts in `data/articles.json` are fictional mock data for development — not the real Kodeks Pracy.

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first indexing run downloads the `BAAI/bge-m3` model (~2 GB) from Hugging Face.

## Usage

Run the pipeline in order:

```bash
# Step 1: Embed articles and build FAISS index
python embed_and_index.py

# Step 2: Interactive semantic search (Polish queries)
python search_cli.py
```

Example query in the CLI:

```
Zapytanie (lub 'exit'): Ile dni urlopu mi przysługuje?
```

## Project structure

```text
nlplaw/
├── embed_and_index.py      # Builds index/kp_index.faiss + kp_metadata.json
├── search_cli.py           # Interactive TOP-5 search CLI
├── rag/
│   ├── config.py           # Paths, model name, TOP_K
│   ├── embeddings.py       # BGE-M3 encode helpers
│   └── index_store.py      # FAISS + metadata I/O
├── data/articles.json      # Article dataset (30 entries)
└── index/
    ├── kp_index.faiss      # FAISS IndexFlatIP (generated)
    └── kp_metadata.json    # Metadata aligned by vector ID (generated)
```

## How it works

- **Embeddings:** Documents are encoded with `BAAI/bge-m3` via `sentence-transformers`. Queries use the BGE retrieval prefix for asymmetric search.
- **Normalization:** Vectors are L2-normalized so that inner product equals cosine similarity.
- **FAISS:** `IndexFlatIP` (inner product) on normalized vectors returns cosine similarity scores.
- **Metadata mapping:** FAISS row `i` maps to `kp_metadata.json[i]`, linking vector IDs back to article title and text.
