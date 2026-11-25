"""Retrieval with two independent filters: a threshold and K.

The threshold runs on its own. Every chunk scoring at or above it is
relevant enough to use; every chunk below it is out, whatever K says.
If nothing clears the threshold, the complaint honestly matches
nothing.

K runs on its own. It decides how many of the surviving chunks are
actually used, in one of two ways:

- fixed:   take exactly the first K survivors.
- dynamic: start from the best survivor and keep taking chunks while
           each next score stays close to the previous one; stop at
           the cliff where scores fall away, or at the maximum K.

Both filters are always on and neither disables the other.

Similarity is 1 - cosine distance, as a fraction (0.508 means 50.8%).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def _hits(collection, query_embedding, pool: int) -> list[dict]:
    total = collection.count()
    if total == 0:
        return []
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(pool, total))
    hits = []
    for chunk_id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0]):
        hits.append({
            "chunk_id": chunk_id, "text": doc,
            "slug": meta["slug"], "title": meta["title"],
            "section": meta["section"],
            "page_start": meta["page_start"], "page_end": meta["page_end"],
            "similarity": round(max(0.0, 1.0 - dist), 4),
        })
    hits.sort(key=lambda h: h["similarity"], reverse=True)
    return hits


def apply_filters(candidates: list[dict], threshold: float, k_mode: str,
                  fixed_k: int, max_k: int,
                  drop_points: float = config.DYNAMIC_DROP_POINTS
                  ) -> list[dict]:
    """First the threshold on its own, then K on its own."""
    survivors = [h for h in candidates
                 if h["similarity"] >= threshold]
    if k_mode == "fixed":
        return survivors[:fixed_k]
    if k_mode == "dynamic":
        if not survivors:
            return []
        used = [survivors[0]]
        for prev, nxt in zip(survivors, survivors[1:]):
            if len(used) >= max_k:
                break
            if prev["similarity"] - nxt["similarity"] > drop_points:
                break
            used.append(nxt)
        return used
    raise ValueError(f"unknown K behaviour: {k_mode}")


def retrieve(collection, embed_fn, query_text: str,
             threshold: float = config.THRESHOLD_DEFAULT,
             k_mode: str = config.K_MODE_DEFAULT,
             fixed_k: int = config.TOP_K_DEFAULT,
             max_k: int = config.MAX_K_DEFAULT) -> dict:
    embedding = embed_fn(query_text)
    candidates = _hits(collection, embedding, config.CANDIDATE_POOL)
    used = apply_filters(candidates, threshold, k_mode, fixed_k, max_k)
    return {"candidates": candidates, "used": used,
            "grounded": len(used) > 0,
            "settings": {"threshold": threshold, "k_mode": k_mode,
                         "fixed_k": fixed_k, "max_k": max_k}}
