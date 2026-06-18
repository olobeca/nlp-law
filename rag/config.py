"""Central configuration for paths, model settings, and search parameters."""

from pathlib import Path

# Project root (parent of the rag/ package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Embedding model
MODEL_NAME = "BAAI/bge-m3"

# BGE-M3 asymmetric retrieval prefix for queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# File paths (relative to project root)
ARTICLES_PATH = PROJECT_ROOT / "data" / "articles.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "index" / "kp_index.faiss"
METADATA_PATH = PROJECT_ROOT / "index" / "kp_metadata.json"

# Search defaults
TOP_K = 5
