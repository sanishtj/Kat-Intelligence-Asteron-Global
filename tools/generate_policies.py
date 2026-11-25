"""Generate the 10 fake Asteron Global policy PDFs from policies-src/*.txt.

Source format:
    TITLE: <policy title>
    SHORT: <slug>
    VERSION: <version line>
    OWNER: <owning team>
    INTRO: <one paragraph shown under the title>

    ## 1. Section heading
    Paragraph text. Lines starting with "- " become bullets.

Output: policies-pdf/<slug>.pdf with Asteron letterhead, footer page
numbers, and real extractable text (so the Phase 2 pipeline can read it
back page by page with PyMuPDF).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,
                                Paragraph, Spacer)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "policies-src"
OUT_DIR = ROOT / "policies-pdf"

BRAND = HexColor("#14467C")
GREY = HexColor("#5A6B7C")
LIGHT = HexColor("#EAF0F6")
DARK = HexColor("#1B2A38")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm


def draw_star(canvas, cx, cy, r):
    canvas.setStrokeColor(HexColor("#FFFFFF"))
    canvas.setLineWidth(2.6)
    canvas.setLineCap(1)
    for angle in (90, 30, 150):
        rad = math.radians(angle)
        dx, dy = r * math.cos(rad), r * math.sin(rad)
        canvas.line(cx - dx, cy - dy, cx + dx, cy + dy)


class PolicyDoc(BaseDocTemplate):
    def __init__(self, filename, policy, **kw):
        super().__init__(filename, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=26 * mm, bottomMargin=18 * mm, **kw)
        self.policy = policy
        frame = Frame(MARGIN, 18 * mm, PAGE_W - 2 * MARGIN,
                      PAGE_H - 26 * mm - 18 * mm, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame],
                                            onPage=self.decorate)])

    def decorate(self, canvas, doc):
        canvas.saveState()
        # Header band: logo block + wordmark left, policy title right.
        canvas.setFillColor(BRAND)
        canvas.roundRect(MARGIN, PAGE_H - 20 * mm, 13 * mm, 13 * mm, 2.5 * mm,
                         stroke=0, fill=1)
        draw_star(canvas, MARGIN + 6.5 * mm, PAGE_H - 13.5 * mm, 4.2 * mm)
        canvas.setFillColor(BRAND)
        canvas.setFont("Helvetica-Bold", 12.5)
        canvas.drawString(MARGIN + 16 * mm, PAGE_H - 12.2 * mm, "ASTERON")
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN + 16.2 * mm, PAGE_H - 16.4 * mm, "G L O B A L")
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 8.5)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 12.2 * mm,
                               self.policy["title"])
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 16.4 * mm,
                               "Internal Use Only")
        canvas.setStrokeColor(BRAND)
        canvas.setLineWidth(0.7)
        canvas.line(MARGIN, PAGE_H - 22 * mm, PAGE_W - MARGIN, PAGE_H - 22 * mm)
        # Footer.
        canvas.setStrokeColor(HexColor("#C9D4DF"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, 10 * mm,
                          f"Asteron Global  |  {self.policy['title']}  |  "
                          f"{self.policy['version']}")
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()


STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19,
                            leading=23, textColor=DARK, spaceAfter=4),
    "meta": ParagraphStyle("meta", fontName="Helvetica", fontSize=9.5,
                           leading=13, textColor=GREY),
    "intro": ParagraphStyle("intro", fontName="Helvetica-Oblique",
                            fontSize=10.5, leading=15, textColor=DARK,
                            alignment=TA_JUSTIFY, spaceBefore=6, spaceAfter=4),
    "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12,
                        leading=15, textColor=BRAND, spaceBefore=11,
                        spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5,
                           leading=15, textColor=DARK, alignment=TA_JUSTIFY,
                           spaceAfter=5),
    "bullet": ParagraphStyle("bullet", parent=ParagraphStyle("b"),
                             fontName="Helvetica", fontSize=10.5, leading=15,
                             textColor=DARK, alignment=TA_LEFT,
                             leftIndent=12, bulletIndent=2, spaceAfter=3),
}


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def parse_source(path: Path) -> dict:
    meta, sections = {}, []
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith(("TITLE:", "SHORT:", "VERSION:", "OWNER:", "INTRO:")):
            key, _, value = line.partition(":")
            meta[key.lower()] = value.strip()
        elif line.startswith("## "):
            current = {"heading": line[3:].strip(), "blocks": []}
            sections.append(current)
        elif line.strip() and current is not None:
            current["blocks"].append(line.strip())
    return {"meta": meta, "sections": sections}


def build_pdf(parsed: dict, out_path: Path) -> None:
    meta = parsed["meta"]
    policy = {"title": meta["title"], "version": meta["version"]}
    doc = PolicyDoc(str(out_path), policy)
    story = [
        Paragraph(esc(meta["title"]), STYLES["title"]),
        Paragraph(f"{esc(meta['version'])} &nbsp;|&nbsp; Owner: "
                  f"{esc(meta['owner'])} &nbsp;|&nbsp; Applies to: all Asteron "
                  f"Global employees, contractors and interns",
                  STYLES["meta"]),
        Paragraph(esc(meta["intro"]), STYLES["intro"]),
        Spacer(1, 4),
    ]
    for section in parsed["sections"]:
        story.append(Paragraph(esc(section["heading"]), STYLES["h"]))
        for block in section["blocks"]:
            if block.startswith("- "):
                story.append(Paragraph(esc(block[2:]), STYLES["bullet"],
                                       bulletText="•"))
            else:
                story.append(Paragraph(esc(block), STYLES["body"]))
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in sorted(SRC_DIR.glob("*.txt")):
        parsed = parse_source(src)
        slug = parsed["meta"]["short"]
        out_path = OUT_DIR / f"{slug}.pdf"
        build_pdf(parsed, out_path)
        manifest.append({"slug": slug, "title": parsed["meta"]["title"],
                         "owner": parsed["meta"]["owner"],
                         "version": parsed["meta"]["version"],
                         "file": out_path.name})
        print(f"built {out_path.name}")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"{len(manifest)} policies written to {OUT_DIR}")


if __name__ == "__main__":
    main()
