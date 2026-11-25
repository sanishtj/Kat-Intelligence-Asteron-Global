"""Mechanics tests for the Kat Intelligence pipeline.

These verify the moving parts without an OpenAI key: extraction keeps
page and section, chunk metadata is correct, the embedder skips text it
has already embedded (fake client), and the Chroma store round-trips
with synthetic vectors.
"""

import json
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))

import config
import embedder
import kb_store
from chunker import chunk_document
from extract import extract_document


@pytest.fixture(scope="module")
def manifest():
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))


def test_extract_keeps_pages_and_sections(manifest):
    meta = next(m for m in manifest if m["slug"] == "anti-bribery-and-corruption")
    data = extract_document(config.POLICIES_DIR / meta["file"], meta)
    assert data["pages"] == 3
    sections = {s["section"] for s in data["segments"]}
    assert "4. Gifts and hospitality you must never accept" in sections
    joined = " ".join(s["text"] for s in data["segments"])
    assert "Internal Use Only" not in joined  # header furniture removed
    assert "paid holiday" in joined


def test_chunk_metadata_points_at_exact_page(manifest):
    meta = next(m for m in manifest if m["slug"] == "conflicts-of-interest")
    data = extract_document(config.POLICIES_DIR / meta["file"], meta)
    chunks = chunk_document(data)
    hit = next(c for c in chunks if "withdraw from every part" in c["text"])
    assert hit["section"] == "5. Supplier and purchasing decisions"
    assert hit["page_start"] == 2 and hit["page_end"] == 2
    assert all(len(c["text"]) <= config.CHUNK_MAX_CHARS + 200
               for c in chunks)


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, model, input):
        self.calls += 1

        class Item:
            def __init__(self, i):
                self.embedding = [float(i + 1)] + [0.0] * (
                    config.EMBEDDING_DIMENSIONS - 1)

        class Resp:
            pass

        resp = Resp()
        resp.data = [Item(i) for i in range(len(input))]
        return resp


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


def test_embedder_skips_already_embedded(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk = {"chunk_id": "doc_chunk000", "slug": "doc", "title": "Doc",
             "section": "1. Purpose", "page_start": 1, "page_end": 1,
             "text": "Asteron Global forbids bribes in every country."}
    (chunks_dir / "doc.jsonl").write_text(json.dumps(chunk) + "\n")
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", tmp_path)
    monkeypatch.setattr(config, "EMBEDDINGS_PATH",
                        tmp_path / "embeddings.jsonl")
    client = FakeClient()
    first = embedder.embed_all(client, 0.02)
    assert first["embedded_now"] == 1 and first["records_total"] == 1
    second = embedder.embed_all(client, 0.02)
    assert second["embedded_now"] == 0 and second["skipped"] == 1
    assert client.embeddings.calls == 1  # no second API call, no re-pay


def test_store_roundtrip_with_synthetic_vectors(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk = {"chunk_id": "doc_chunk000", "slug": "doc", "title": "Doc",
             "section": "4. Gifts", "page_start": 2, "page_end": 2,
             "text": "Never accept a paid holiday from a vendor."}
    (chunks_dir / "doc.jsonl").write_text(json.dumps(chunk) + "\n")
    emb_path = tmp_path / "embeddings.jsonl"
    emb_path.write_text(json.dumps({
        "chunk_id": "doc_chunk000",
        "text_hash": embedder.text_hash(chunk["text"]),
        "embedding": [1.0] + [0.0] * (config.EMBEDDING_DIMENSIONS - 1),
    }) + "\n")
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "EMBEDDINGS_PATH", emb_path)
    result = kb_store.store_all(chroma_dir=tmp_path / "chroma")
    assert result["stored"] == 1 and result["collection_count"] == 1
    coll = kb_store.get_collection(tmp_path / "chroma")
    found = coll.query(
        query_embeddings=[[1.0] + [0.0] * (config.EMBEDDING_DIMENSIONS - 1)],
        n_results=1)
    meta = found["metadatas"][0][0]
    assert meta["section"] == "4. Gifts" and meta["page_start"] == 2
