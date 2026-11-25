"""Step 5 - Store: upsert embedded chunks into the ChromaDB folder.

The store is a folder on disk (config.CHROMA_DIR), exactly as in the
bootcamp project. Chunk metadata carries the citation: document title,
section, and the page the chunk starts and ends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from embedder import load_chunks, text_hash

BATCH_SIZE = 200


def get_collection(chroma_dir: Path | None = None):
    client = chromadb.PersistentClient(path=str(chroma_dir or config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def store_all(chroma_dir: Path | None = None) -> dict:
    chunks = {c["chunk_id"]: c for c in load_chunks()}
    if not config.EMBEDDINGS_PATH.exists():
        raise SystemExit("No embeddings found. Run the embed step first "
                         "(it needs the OpenAI key).")
    by_hash = {}
    for line in config.EMBEDDINGS_PATH.read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_hash[rec["text_hash"]] = rec
    ready = []
    missing = 0
    for chunk in chunks.values():
        rec = by_hash.get(text_hash(chunk["text"]))
        if rec is None:
            missing += 1
            continue
        ready.append((chunk, rec["embedding"]))
    collection = get_collection(chroma_dir)
    for start in range(0, len(ready), BATCH_SIZE):
        batch = ready[start:start + BATCH_SIZE]
        collection.upsert(
            ids=[c["chunk_id"] for c, _ in batch],
            embeddings=[e for _, e in batch],
            documents=[c["text"] for c, _ in batch],
            metadatas=[{
                "slug": c["slug"], "title": c["title"],
                "section": c["section"],
                "page_start": c["page_start"], "page_end": c["page_end"],
            } for c, _ in batch],
        )
    return {"stored": len(ready), "missing_embeddings": missing,
            "collection_count": collection.count()}


if __name__ == "__main__":
    try:
        result = store_all()
    except SystemExit as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: storing failed ({type(exc).__name__}): {exc}")
        raise SystemExit(1)
    print(f"stored: {result['stored']}, "
          f"missing embeddings: {result['missing_embeddings']}, "
          f"collection count: {result['collection_count']}")
