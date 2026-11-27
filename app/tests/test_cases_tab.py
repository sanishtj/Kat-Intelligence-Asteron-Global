"""App-level test for the Kat Intelligence cases tab.

Boots the real app with AppTest against a tiny fake Chroma store
(known vectors, the pattern from test_engine.py) and a fake AI
client patched in where app.py builds its client, so no key and
no network are involved. Clicking a case button must show the
case and its matched policies, and must run the search only
once: the outcome is cached per case.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import chromadb
import openai
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent
ENGINE = APP / "engine"
sys.path.insert(0, str(ENGINE))

import config

DIM = 1536


def vec(similarity: float) -> list[float]:
    return [similarity, math.sqrt(1 - similarity ** 2)] + [0.0] * (DIM - 2)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    path = tmp_path_factory.mktemp("chroma")
    client = chromadb.PersistentClient(path=str(path))
    coll = client.get_or_create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    coll.upsert(
        ids=["whp_chunk001", "whp_chunk002", "abc_chunk001"],
        embeddings=[vec(0.90), vec(0.85), vec(0.30)],
        documents=[
            "Unwanted personal messages that continue after the "
            "person has said no are harassment.",
            "Report harassment to the committee; retaliation "
            "against a reporter is itself misconduct.",
            "Never accept a gift from a company that is bidding.",
        ],
        metadatas=[
            {"slug": "workplace-harassment-prevention",
             "title": "Workplace Harassment Prevention Policy",
             "section": "2. What counts as harassment",
             "page_start": 1, "page_end": 1},
            {"slug": "workplace-harassment-prevention",
             "title": "Workplace Harassment Prevention Policy",
             "section": "5. Reporting",
             "page_start": 3, "page_end": 3},
            {"slug": "anti-bribery-and-corruption",
             "title": "Anti-Bribery and Corruption Policy",
             "section": "4. Gifts",
             "page_start": 2, "page_end": 2},
        ],
    )
    return path


class _Embeddings:
    @staticmethod
    def create(model=None, input=None):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vec(1.0))],
            usage=SimpleNamespace(total_tokens=7))


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **_kw):
        self._owner.chat_calls += 1

        def chunk(text, usage=None):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text))],
                usage=usage)

        return iter([
            chunk("The Workplace Harassment Prevention Policy "
                  "covers unwanted messages that continue after "
                  "the person has said no.\n"),
            chunk("USED: P1"),
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5,
                total_tokens=15)),
        ])


class FakeClient:
    """Stands in for the OpenAI and Azure clients alike: the app
    resolves both names from the openai module at call time."""

    def __init__(self):
        self.chat_calls = 0
        self.embeddings = _Embeddings()
        self.chat = SimpleNamespace(
            completions=_Completions(self))


@pytest.fixture()
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda **_kw: client)
    monkeypatch.setattr(openai, "AzureOpenAI", lambda **_kw: client)
    return client


def boot(store, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(store))
    return AppTest.from_file(str(APP / "app.py"),
                             default_timeout=60).run()


def markdown_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_case_click_shows_case_and_matches(store, monkeypatch,
                                           fake_client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = boot(store, monkeypatch)
    assert not at.exception
    button = next(b for b in at.button
                  if b.label == "Messages That Will Not Stop")
    at = button.click().run()
    assert not at.exception
    # The full case, with its reported-by line.
    assert "dread seeing him at work" in markdown_text(at)
    assert any("Reported by a customer support agent, Operations"
               in c.value for c in at.caption)
    # The short answer and the matched-policy area.
    assert "covers unwanted messages" in markdown_text(at)
    assert "Policies that match this case" in markdown_text(at)
    assert any("Workplace Harassment Prevention Policy" in b.label
               for b in at.button)
    assert fake_client.chat_calls == 1
    # A second click is served from the cache: no new model call.
    button = next(b for b in at.button
                  if b.label == "Messages That Will Not Stop")
    at = button.click().run()
    assert not at.exception
    assert fake_client.chat_calls == 1


def test_app_boots_in_azure_mode(store, monkeypatch, fake_client):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT",
                       "https://fake-resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = boot(store, monkeypatch)
    assert not at.exception
    assert any(b.label == "Messages That Will Not Stop"
               for b in at.button)
