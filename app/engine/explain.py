"""Generation: a grounded explanation of why the matched policies fit.

The complaint is never sent to the model alone; it always travels with
the retrieved passages, each labelled with its policy's name, section
and page, so the model can cite policies by name. When nothing matched, the
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
            f"[P{i}] {hit['title']} — {hit['section']} "
            f"(page {hit['page_start']})\n{hit['text']}")
    return "\n\n".join(parts)


_USED_LINE = None  # compiled on first use, see parse_used_marker


def parse_used_marker(text: str) -> tuple[str, list[int] | None]:
    """Split the model's reply into (answer, used indices).

    The reply contract ends with a line "USED: P1, P3" naming
    the passages the answer rests on, or "USED: none". Returns
    the reply with that line removed and the 0-based indices it
    named, or (text, None) when no valid USED line is present,
    so the caller can fall back to treating every retrieved
    passage as used."""
    import re

    global _USED_LINE
    if _USED_LINE is None:
        _USED_LINE = re.compile(r"(?m)^USED:\s*(.+?)\s*$")
    if not text:
        return text, None
    match = None
    for match in _USED_LINE.finditer(text):
        pass
    if match is None:
        return text, None
    tail = match.group(1).rstrip(".")
    if tail.lower() == "none":
        return text[:match.start()].rstrip(), []
    indices = []
    for part in tail.split(","):
        m = re.fullmatch(r"P(\d+)", part.strip().rstrip("."))
        if m is None:
            return text, None
        indices.append(int(m.group(1)) - 1)
    return text[:match.start()].rstrip(), indices


def build_messages(complaint: str, used: list[dict],
                   history: list[dict] | None = None) -> list[dict]:
    """The messages for one answer. With `history` (the chat's
    earlier exchanges, as role/content dicts), the conversation
    rides between the system prompt and the final message, so a
    follow-up can be understood against what came before. The
    history is context for understanding the message, never a
    source of facts: the grounding addendum says so, and the
    passages in the final message stay the only evidence."""
    user = (f"Message from an employee:\n{complaint}\n\n"
            f"Policy passages retrieved:\n{build_context(used)}")
    if not history:
        return [{"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    system = (config.SYSTEM_PROMPT + "\n\nThe conversation so far "
              "is included only so you can understand a follow-up "
              "message. It is context, never a source: every fact "
              "in your answer must come from the policy passages "
              "in the final message.")
    return ([{"role": "system", "content": system}]
            + [{"role": h["role"], "content": h["content"]}
               for h in history]
            + [{"role": "user", "content": user}])


def explain(client, complaint: str,
            used: list[dict]) -> tuple[str, list[dict]] | None:
    """Return (answer, used_passages), or None when there is nothing
    to ground the answer in (the caller then shows NO_MATCH_MESSAGE;
    no API call is made). used_passages is the subset the model says
    its answer rests on, read from the reply's USED line: when that
    line says none, the answer is NOT_COVERED_MESSAGE and the subset
    is empty; without a valid USED line, every retrieved passage
    counts as used (the fallback). The USED line itself is never
    part of the returned answer.
    """
    if not used:
        return None
    response = client.chat.completions.create(
        model=config.CHAT_MODEL, temperature=0.2,
        messages=build_messages(complaint, used))
    content = response.choices[0].message.content
    answer, indices = parse_used_marker(content)
    if indices is None:
        return content, used
    if not indices:
        return config.NOT_COVERED_MESSAGE, []
    subset = [used[i] for i in indices if 0 <= i < len(used)]
    if not subset:
        return answer, used
    return answer, subset
