from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "BAAI/bge-m3"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ARTICLES_PATH = PROJECT_ROOT / "data" / "articles.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "index" / "kp_index.faiss"
METADATA_PATH = PROJECT_ROOT / "index" / "kp_metadata.json"

TOP_K = 5

RERANKER_ENABLED_DEFAULT = True
RERANKER_CANDIDATES_DEFAULT = 15
RERANKER_MODEL_NAME = "nreimers/mmarco-mMiniLMv2-L6-H384-v1"
RERANKER_MAX_CHARS = 2000

RERANKER_MODELS = {
    "mmarco-L6 (multilingual, domyślny)": "nreimers/mmarco-mMiniLMv2-L6-H384-v1",
    "mmarco-L12 (multilingual)": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "ms-marco-L6 (szybki, EN)": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-L12 (dokładniejszy, EN)": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}
