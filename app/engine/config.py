"""Settings for the Kat Intelligence playground engine.

Defaults are starting points only: Phase 5's test pass sets the shipped
defaults from measured scores, the same discipline as the bootcamp
chatbot (Top-K 10, threshold 35% were earned, not guessed).
"""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent

COLLECTION_NAME = "asteron_policies"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def chat_temperature(model: str):
    """The temperature to send with a chat call, or None to
    send none. GPT-5 models accept only their default
    temperature and reject a sent one, so for them the app
    sends nothing and the model uses its default."""
    if model.strip().lower().startswith("gpt-5"):
        return None
    return 0.2

# Retrieval defaults (similarity is a fraction: 0.35 means 35%).
# Two independent filters, both always on: the threshold decides which
# chunks are relevant enough, K decides how many of those are used.
THRESHOLD_DEFAULT = 0.35            # the relevance floor
K_MODE_DEFAULT = "dynamic"          # dynamic | fixed
TOP_K_DEFAULT = 5                   # fixed K: passages used
MAX_K_DEFAULT = 5                   # dynamic K: cap on passages used
DYNAMIC_DROP_POINTS = 0.08          # dynamic K: stop when the score
                                    # falls more than this below the
                                    # previous chunk's score
CANDIDATE_POOL = 25                 # chunks fetched before filtering

NO_MATCH_MESSAGE = (
    "No Asteron policy matched this closely enough to name one "
    "reliably. That can be the right answer: not everything at work is "
    "covered by a policy. If you believe a policy should cover it, the "
    "Speak-Up and Whistleblowing Policy explains how to raise it."
)

# The reply contract (see the system prompt): the model ends its
# reply with a USED line naming the passages its answer actually
# rests on, or "USED: none" when the passages that passed the
# filters do not answer the message (vector search always returns
# the closest chunks, however irrelevant). The app parses that
# line, never shows it, and decides from it: no used passages
# means the not-covered state below, with the checked passages
# listed collapsed and without match percentages, so a
# checked-but-irrelevant passage is never dressed up as a source;
# some used passages means only those are shown as sources.
NOT_COVERED_MESSAGE = (
    "The closest passages were checked, and none of them cover "
    "this, so no answer was given."
)

SYSTEM_PROMPT = (
    "You are Kat Intelligence, an assistant that answers from a "
    "company's own policy documents. You are given a message from an "
    "employee and passages from the company's policies. The message "
    "may be a question, a request for information, or a report of a "
    "problem. Treat it as what it actually is: answer a question as "
    "a question, and only describe a message as a complaint or a "
    "report when the person is actually reporting a problem. Never "
    "call a question a complaint. Using ONLY the policy passages "
    "provided, explain in plain everyday language what the policies "
    "say that bears on the message, naming the specific rule in each "
    "passage that connects to it. When you mention a policy, refer "
    "to it by its name, for example 'the Workplace Harassment "
    "Prevention Policy', adding its section where that helps. Never "
    "refer to a passage as 'Source 1', 'Source 2' or by any "
    "numbered label: those labels are not part of your answer. "
    "Do not invent rules, page numbers or policies that are not in "
    "the passages. Each passage is labelled with a code like "
    "[P1]: those codes are plumbing for the app, not citations, "
    "so never mention them in your answer. Finish your reply "
    "with one last line in exactly this form: USED: P1, P3 — "
    "the codes of the passages your answer actually rests on, "
    "and no others. If none of the passages really answer the "
    "message, make that line your whole reply: USED: none, "
    "with nothing before or after it. Judge coverage by the "
    "topic of the message, not by loose relatedness: a passage "
    "that only explains how to report or raise an issue does "
    "not cover a question about a topic no passage addresses, "
    "so in that case your whole reply is USED: none even when "
    "reporting passages were retrieved. When a passage does "
    "address the topic, reporting steps from the passages may "
    "be part of the answer. Keep it short: a few "
    "sentences per policy at most."
)


def chroma_dir() -> Path:
    """Where the ChromaDB folder lives, in priority order."""
    env = os.environ.get("CHROMA_PATH")
    if env:
        return Path(env)
    packaged = APP_DIR / "data" / "chroma"
    if packaged.exists():
        return packaged
    return PROJECT_DIR / "pipeline" / "storage" / "chroma"


def policies_dir() -> Path:
    packaged = APP_DIR / "data" / "policies"
    if packaged.exists():
        return packaged
    return PROJECT_DIR / "policies-pdf"
