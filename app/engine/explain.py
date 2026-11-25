"""Generation: a grounded explanation of why the matched policies fit.

The complaint is never sent to the model alone; it always travels with
the retrieved passages, labelled [Source N]. When nothing matched, the
caller shows the no-match message and the model is never called - the
same defence in depth as the bootcamp chatbot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def build_context(used: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(used, start=1):
        parts.append(
            f"[Source {i}] {hit['title']} - {hit['section']} "
            f"(page {hit['page_start']})\n{hit['text']}")
    return "\n\n".join(parts)


def build_messages(complaint: str, used: list[dict]) -> list[dict]:
    user = (f"Message from an employee:\n{complaint}\n\n"
            f"Policy passages retrieved:\n{build_context(used)}")
    return [{"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def explain(client, complaint: str, used: list[dict]) -> str | None:
    """Return the explanation, or None when there is nothing to ground
    it in (the caller then shows NO_MATCH_MESSAGE; no API call is made).
    """
    if not used:
        return None
    response = client.chat.completions.create(
        model=config.CHAT_MODEL, temperature=0.2,
        messages=build_messages(complaint, used))
    return response.choices[0].message.content
