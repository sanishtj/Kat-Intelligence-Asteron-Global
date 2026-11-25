"""Step 2 - Clean: conservative cleanup of extracted segments.

Same philosophy as the bootcamp pipeline: normalise whitespace, collapse
blank runs, drop segments that end up empty. Nothing is rewritten.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_all() -> list[dict]:
    config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    paths = sorted(config.EXTRACTED_DIR.glob("*.json"))
    if not paths:
        print("Nothing to clean: no extracted files found. Run the "
              "Extract step first (button 1).")
        return results
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        kept = []
        for seg in data["segments"]:
            text = clean_text(seg["text"])
            if text:
                kept.append({**seg, "text": text})
        data["segments"] = kept
        total_chars = sum(len(s["text"]) for s in kept)
        data["cleaned_chars"] = total_chars
        data["status"] = ("EMPTY" if total_chars < config.MIN_CLEANED_CHARS
                          else "CLEANED")
        out = config.CLEANED_DIR / path.name
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        results.append(data)
        print(f"cleaned {path.name}: {len(kept)} segments, "
              f"{total_chars} chars [{data['status']}]")
    return results


if __name__ == "__main__":
    try:
        clean_all()
    except Exception as exc:
        print(f"ERROR: cleaning failed ({type(exc).__name__}): {exc}")
        raise SystemExit(1)
