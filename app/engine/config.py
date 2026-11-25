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
    "the passages. If the passages do not really answer the message, "
    "say so plainly. Keep it short: a few sentences per policy at "
    "most."
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
