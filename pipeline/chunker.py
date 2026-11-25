"""Step 3 - Chunk: pack cleaned segments into retrieval-sized chunks.

Structure first, like the bootcamp chunker: a chunk never crosses a
section boundary. Each chunk records the document, the section, and the
page it starts and ends on, so a match can open the PDF at the exact
page. Chunks are capped at CHUNK_MAX_CHARS with a small overlap carried
from the previous chunk's tail; a short final chunk is merged into its
neighbour instead of standing alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def _overlap(text: str) -> str:
    if len(text) <= config.CHUNK_OVERLAP_CHARS:
        return text
    tail = text[-config.CHUNK_OVERLAP_CHARS:]
    cut = tail.find(" ")
    return tail[cut + 1:] if cut > 0 else tail


def chunk_document(data: dict) -> list[dict]:
    chunks: list[dict] = []
    # Group segments by section, preserving order.
    sections: list[tuple[str, list[dict]]] = []
    for seg in data["segments"]:
        if not sections or sections[-1][0] != seg["section"]:
            sections.append((seg["section"], []))
        sections[-1][1].append(seg)

    for section, segs in sections:
        current: list[dict] = []
        current_len = 0

        def flush():
            nonlocal current, current_len
            if not current:
                return
            text = "\n".join(s["text"] for s in current)
            chunks.append({
                "chunk_id": f"{data['slug']}_chunk{len(chunks):03d}",
                "slug": data["slug"], "title": data["title"],
                "section": section,
                "page_start": current[0]["page"],
                "page_end": current[-1]["page"],
                "text": text,
            })
            current, current_len = [], 0

        for seg in segs:
            seg_len = len(seg["text"]) + 1
            if current and current_len + seg_len > config.CHUNK_MAX_CHARS:
                tail_text = "\n".join(s["text"] for s in current)
                flush()
                if config.CHUNK_OVERLAP_CHARS:
                    current = [{"page": seg["page"],
                                "text": _overlap(tail_text)}]
                    current_len = len(current[0]["text"])
            current.append(seg)
            current_len += seg_len
        flush()

    # Merge a too-short final chunk of the document into its neighbour.
    if len(chunks) >= 2 and len(chunks[-1]["text"]) < config.CHUNK_MIN_TAIL_CHARS:
        last = chunks.pop()
        prev = chunks[-1]
        prev["text"] = prev["text"] + "\n" + last["text"]
        prev["page_end"] = last["page_end"]
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{data['slug']}_chunk{i:03d}"
    return chunks


def chunk_all() -> list[dict]:
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    paths = sorted(config.CLEANED_DIR.glob("*.json"))
    if not paths:
        print("Nothing to chunk: no cleaned files found. Run the "
              "Clean step first (button 2).")
        return all_chunks
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "EMPTY":
            print(f"skipped {path.name}: EMPTY")
            continue
        chunks = chunk_document(data)
        out = config.CHUNKS_DIR / f"{data['slug']}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk) + "\n")
        all_chunks.extend(chunks)
        print(f"chunked {path.name}: {len(chunks)} chunks")
    print(f"TOTAL chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    try:
        chunk_all()
    except Exception as exc:
        print(f"ERROR: chunking failed ({type(exc).__name__}): {exc}")
        raise SystemExit(1)
