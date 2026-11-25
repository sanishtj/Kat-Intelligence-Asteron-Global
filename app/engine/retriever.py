"""Retrieval with three selectable modes, for the playground controls.

- fixed_k:   take the top K chunks, whatever their scores. Shows why a
             bare Top-K can drag in weak matches.
- threshold: take every chunk at or above the relevance threshold, up
             to a maximum K. The honest mode: it can return nothing.
- dynamic_k: start from the best chunk and keep taking chunks while
             each next score stays close to the previous one; stop at
             the cliff where scores fall away, or at max K. The top
             chunk must still clear the threshold.

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


def apply_mode(candidates: list[dict], mode: str, top_k: int,
               threshold: float, max_k: int,
               drop_points: float = config.DYNAMIC_DROP_POINTS
               ) -> list[dict]:
    if mode == "fixed_k":
        return candidates[:top_k]
    if mode == "threshold":
        return [h for h in candidates
                if h["similarity"] >= threshold][:max_k]
    if mode == "dynamic_k":
        if not candidates or candidates[0]["similarity"] < threshold:
            return []
        used = [candidates[0]]
        for prev, nxt in zip(candidates, candidates[1:]):
            if len(used) >= max_k:
                break
            if prev["similarity"] - nxt["similarity"] > drop_points:
                break
            used.append(nxt)
        return used
    raise ValueError(f"unknown retrieval mode: {mode}")


def retrieve(collection, embed_fn, query_text: str, mode: str,
             top_k: int = config.TOP_K_DEFAULT,
             threshold: float = config.THRESHOLD_DEFAULT,
             max_k: int = config.MAX_K_DEFAULT) -> dict:
    embedding = embed_fn(query_text)
    candidates = _hits(collection, embedding, config.CANDIDATE_POOL)
    used = apply_mode(candidates, mode, top_k, threshold, max_k)
    return {"candidates": candidates, "used": used,
            "grounded": len(used) > 0, "mode": mode}
