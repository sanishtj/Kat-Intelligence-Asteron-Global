"""OpenAI key handling, same pattern as Sanish's bootcamp project.

The key is read from the environment first, else from this project's own
.streamlit/secrets.toml. It is never printed in full, never stored in the
knowledge base, and never committed (secrets.toml is git-ignored).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

SECRETS_PATH = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"


def get_api_key() -> str | None:
    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env.strip()
    if SECRETS_PATH.exists():
        try:
            data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
        key = data.get("OPENAI_API_KEY")
        if key:
            return str(key).strip()
    return None


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:3]}...{key[-4:]}"


def key_diagnostics() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "key found in environment variable OPENAI_API_KEY"
    if not SECRETS_PATH.exists():
        return f"no key: {SECRETS_PATH} does not exist"
    try:
        data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"no key: {SECRETS_PATH} could not be parsed ({exc})"
    if not data.get("OPENAI_API_KEY"):
        return f"no key: {SECRETS_PATH} has no OPENAI_API_KEY entry"
    return f"key found in {SECRETS_PATH}"
