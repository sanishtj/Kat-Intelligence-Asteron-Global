"""Configuration for the Kat Intelligence pipeline (Asteron PoC).

The pipeline turns tenant policy PDFs into a ChromaDB folder:
    extract -> clean -> chunk -> embed -> store
Chunk settings match the values Sanish validated in his bootcamp project.
"""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent

# Source PDFs (the tenant's uploaded policies live here).
POLICIES_DIR = PROJECT_DIR / "policies-pdf"
MANIFEST_PATH = POLICIES_DIR / "manifest.json"

# Working storage, one folder per step.
STORAGE_DIR = PIPELINE_DIR / "storage"
EXTRACTED_DIR = STORAGE_DIR / "extracted"
CLEANED_DIR = STORAGE_DIR / "cleaned"
CHUNKS_DIR = STORAGE_DIR / "chunks"
EMBEDDINGS_DIR = STORAGE_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.jsonl"
CHROMA_DIR = STORAGE_DIR / "chroma"

COLLECTION_NAME = "asteron_policies"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
# Default only: the price is an input (env EMBEDDING_PRICE_USD), as in the
# bootcamp dashboard, never a hidden constant.
EMBEDDING_PRICE_USD_PER_1M_DEFAULT = 0.02

CHUNK_MAX_CHARS = 1600
CHUNK_OVERLAP_CHARS = 150
CHUNK_MIN_TAIL_CHARS = 100
MIN_CLEANED_CHARS = 50
