"""Step 4 - Embed: turn chunks into vectors with text-embedding-3-small.

Same discipline as the bootcamp embedder: tiktoken counts tokens before
any API call so cost is known up front; a chunk whose text hash already
has an embedding is skipped, so re-runs never pay twice for the same
text. Output: storage/embeddings/embeddings.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

BATCH_SIZE = 100
_enc = tiktoken.get_encoding("cl100k_base")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(config.CHUNKS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_existing() -> dict[str, dict]:
    existing = {}
    if config.EMBEDDINGS_PATH.exists():
        for line in config.EMBEDDINGS_PATH.read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["text_hash"]] = rec
    return existing


def estimate(chunks: list[dict], price_per_1m: float) -> dict:
    existing = load_existing()
    todo = [c for c in chunks if text_hash(c["text"]) not in existing]
    tokens = sum(count_tokens(c["text"]) for c in todo)
    return {"chunks_total": len(chunks), "chunks_to_embed": len(todo),
            "chunks_skipped": len(chunks) - len(todo), "tokens": tokens,
            "est_cost_usd": tokens / 1_000_000 * price_per_1m}


def embed_all(client, price_per_1m: float) -> dict:
    chunks = load_chunks()
    existing = load_existing()
    todo = [c for c in chunks if text_hash(c["text"]) not in existing]
    tokens = 0
    records = list(existing.values())
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[c["text"] for c in batch])
        for chunk, item in zip(batch, response.data):
            tokens += count_tokens(chunk["text"])
            records.append({
                "chunk_id": chunk["chunk_id"],
                "text_hash": text_hash(chunk["text"]),
                "embedding": item.embedding,
            })
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with config.EMBEDDINGS_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return {"embedded_now": len(todo),
            "skipped": len(chunks) - len(todo),
            "tokens_now": tokens,
            "cost_now_usd": tokens / 1_000_000 * price_per_1m,
            "records_total": len(records)}


def run(price_per_1m: float) -> dict:
    from openai import OpenAI

    import api_key

    key = api_key.get_api_key()
    if not key:
        raise SystemExit(f"Cannot embed: {api_key.key_diagnostics()}")
    print(f"OpenAI key: {api_key.mask_key(key)}")
    chunks = load_chunks()
    est = estimate(chunks, price_per_1m)
    print(f"chunks: {est['chunks_total']} total, "
          f"{est['chunks_to_embed']} to embed, "
          f"{est['chunks_skipped']} already embedded")
    print(f"estimated tokens: {est['tokens']}, "
          f"estimated cost: ${est['est_cost_usd']:.4f}")
    result = embed_all(OpenAI(api_key=key), price_per_1m)
    print(f"embedded now: {result['embedded_now']} "
          f"({result['tokens_now']} tokens, ${result['cost_now_usd']:.4f}); "
          f"records total: {result['records_total']}")
    return result


if __name__ == "__main__":
    import os
    price = float(os.environ.get(
        "EMBEDDING_PRICE_USD", config.EMBEDDING_PRICE_USD_PER_1M_DEFAULT))
    run(price)
