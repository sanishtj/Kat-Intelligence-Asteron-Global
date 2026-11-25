"""Render a policy PDF page as an image with the matched passage lit up.

PyMuPDF searches the page for a distinctive run of words from the chunk
(the chunk text was cleaned, so an exact long match can fail; shorter
windows are tried in turn). The highlight is drawn on an in-memory copy
and the page is rendered to PNG bytes for display. If no window is
found, the plain page is rendered and the result says so honestly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ZOOM = 2.0  # ~150 dpi: crisp on screen, small enough to ship


def _search_windows(text: str) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    windows = []
    for size in (10, 7, 5):
        if len(words) >= size:
            windows.append(" ".join(words[:size]))
    # Also try a window from the middle: a chunk may open with overlap
    # text carried from the previous chunk.
    if len(words) >= 14:
        windows.append(" ".join(words[7:14]))
    return windows


def render_highlighted_page(pdf_path: Path, page_number: int,
                            chunk_text: str) -> dict:
    doc = pymupdf.open(pdf_path)
    page = doc[page_number - 1]
    found = False
    for window in _search_windows(chunk_text):
        rects = page.search_for(window)
        if rects:
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 0.85, 0.2))
                annot.update()
            found = True
            break
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM))
    return {"png": pix.tobytes("png"), "page": page_number,
            "highlighted": found, "pages_total": len(doc)}


def render_for_hit(hit: dict) -> dict:
    pdf_path = config.policies_dir() / f"{hit['slug']}.pdf"
    return render_highlighted_page(pdf_path, hit["page_start"],
                                   hit["text"])
