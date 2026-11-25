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
    for hit in used:
        parts.append(
            f"[{hit['title']} — {hit['section']} "
            f"(page {hit['page_start']})]\n{hit['text']}")
    return "\n\n".join(parts)


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
