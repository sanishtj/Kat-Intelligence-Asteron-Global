"""Tests for the Kat Intelligence playground engine (no OpenAI key).

Retrieval modes are tested against a tiny Chroma collection whose
vectors sit at known cosine similarities to the query. The highlight
renderer is tested against the real generated policy PDFs. The
explainer is tested for its contract: no matches, no model call.
"""

import json
import math
import sys
from pathlib import Path

import chromadb
import pytest

ENGINE = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE))

import config
import explain
import highlight
import retriever

DIM = 1536


def vec(similarity: float) -> list[float]:
    return [similarity, math.sqrt(1 - similarity ** 2)] + [0.0] * (DIM - 2)


@pytest.fixture()
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = client.get_or_create_collection(
        "test", metadata={"hnsw:space": "cosine"})
    sims = [0.90, 0.85, 0.60, 0.30]
    coll.upsert(
        ids=[f"c{i}" for i in range(4)],
        embeddings=[vec(s) for s in sims],
        documents=[f"chunk {i}" for i in range(4)],
        metadatas=[{"slug": "doc", "title": "Doc", "section": f"{i}. S",
                     "page_start": 1, "page_end": 1} for i in range(4)],
    )
    return coll


def run_mode(collection, mode, **kw):
    return retriever.retrieve(collection, lambda _q: vec(1.0), "anything",
                              mode, **kw)


def test_fixed_k_takes_top_k_regardless(collection):
    out = run_mode(collection, "fixed_k", top_k=2)
    assert [h["chunk_id"] for h in out["used"]] == ["c0", "c1"]
    assert out["grounded"] is True


def test_threshold_filters_and_can_return_nothing(collection):
    out = run_mode(collection, "threshold", threshold=0.50, max_k=5)
    assert [h["chunk_id"] for h in out["used"]] == ["c0", "c1", "c2"]
    out = run_mode(collection, "threshold", threshold=0.95, max_k=5)
    assert out["used"] == [] and out["grounded"] is False


def test_dynamic_k_stops_at_the_cliff(collection):
    out = run_mode(collection, "dynamic_k", threshold=0.35, max_k=5)
    assert [h["chunk_id"] for h in out["used"]] == ["c0", "c1"]
    out = run_mode(collection, "dynamic_k", threshold=0.95, max_k=5)
    assert out["used"] == []


def _real_chunk(slug: str, needle: str) -> dict:
    path = (config.PROJECT_DIR / "pipeline" / "storage" / "chunks"
            / f"{slug}.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        if needle in chunk["text"]:
            return chunk
    raise AssertionError(f"chunk not found: {slug} / {needle}")


def test_highlight_finds_passage_on_page_one():
    chunk = _real_chunk("anti-bribery-and-corruption", "paid holiday")
    out = highlight.render_for_hit(chunk)
    assert out["png"][:4] == b"\x89PNG"
    assert out["page"] == 1 and out["highlighted"] is True


def test_highlight_finds_passage_on_page_two():
    chunk = _real_chunk("conflicts-of-interest", "withdraw from every part")
    out = highlight.render_for_hit(chunk)
    assert out["page"] == 2 and out["highlighted"] is True


class ExplodingClient:
    class chat:
        class completions:
            @staticmethod
            def create(**_kw):
                raise AssertionError("model must not be called")


def test_explain_never_calls_model_without_matches():
    assert explain.explain(ExplodingClient(), "anything", []) is None


def test_build_messages_carries_complaint_and_sources():
    used = [{"title": "Anti-Bribery and Corruption Policy",
             "section": "4. Gifts", "page_start": 1,
             "text": "Never accept a paid holiday."}]
    messages = explain.build_messages("A vendor offered me a trip.", used)
    assert "vendor offered me a trip" in messages[1]["content"]
    assert "[Source 1]" in messages[1]["content"]
    assert "page 1" in messages[1]["content"]
