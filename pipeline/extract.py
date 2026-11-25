"""Step 1 - Extract: read each policy PDF page by page with PyMuPDF.

Produces storage/extracted/<slug>.json: a list of segments in reading
order, each {page, section, text}. The page number and the section
heading are captured here and carried through every later step, because
the case panel must open the PDF at the exact page a chunk came from.

Header and footer blocks are dropped by position (they repeat on every
page and would pollute the chunks). No LLM is involved in extraction.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

HEADING_RE = re.compile(r"^(\d{1,2})\.\s+(.+)$")
HEADER_MAX_Y = 64      # points from the top of an A4 page: the letterhead
FOOTER_MIN_Y = 798     # points: the footer rule and line


def _blocks(page) -> list[dict]:
    out = []
    for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks"):
        if btype != 0:
            continue
        text = text.strip()
        if not text:
            continue
        if y1 <= HEADER_MAX_Y or y0 >= FOOTER_MIN_Y:
            continue  # letterhead / footer furniture
        out.append({"y": y0, "text": text})
    out.sort(key=lambda b: b["y"])
    return out


def extract_document(pdf_path: Path, meta: dict) -> dict:
    doc = pymupdf.open(pdf_path)
    segments: list[dict] = []
    section = "Introduction"
    seen_title_block = False
    for page_no, page in enumerate(doc, start=1):
        for block in _blocks(page):
            text = " ".join(block["text"].split("\n")).strip()
            if not seen_title_block and page_no == 1:
                # First-page title block: title + meta line are metadata,
                # not content. The intro paragraph that follows is kept.
                if text == meta["title"] or text.startswith("Version"):
                    if text.startswith("Version"):
                        seen_title_block = True
                    continue
            m = HEADING_RE.match(text)
            if m and len(text) < 90:
                section = f"{m.group(1)}. {m.group(2)}"
                continue
            segments.append({"page": page_no, "section": section,
                             "text": text})
    return {"slug": meta["slug"], "title": meta["title"],
            "owner": meta["owner"], "version": meta["version"],
            "pages": len(doc), "segments": segments}


def extract_all() -> list[dict]:
    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for meta in manifest:
        pdf_path = config.POLICIES_DIR / meta["file"]
        data = extract_document(pdf_path, meta)
        out = config.EXTRACTED_DIR / f"{meta['slug']}.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        results.append(data)
        print(f"extracted {meta['file']}: {data['pages']} pages, "
              f"{len(data['segments'])} segments")
    return results


if __name__ == "__main__":
    try:
        extract_all()
    except FileNotFoundError as exc:
        print(f"ERROR: a file is missing: {exc.filename or exc}. "
              "Check that the policies folder holds the PDFs and "
              "manifest.json.")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: extraction failed ({type(exc).__name__}): "
              f"{exc}")
        raise SystemExit(1)
