"""Kat Intelligence - Asteron Global playground (the public app).

Home is the playground: describe a workplace complaint as if you were
an Asteron employee, and Kat Intelligence finds the company policies
that match it, explains why, and opens the policy PDF at the exact
page, with the passage highlighted.

Tabs: Playground - Policies - How it was built - Retrieval Lab.
"""

import json
import os
import sys
import tomllib
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ENGINE_DIR = APP_DIR / "engine"
sys.path.insert(0, str(ENGINE_DIR))

import chromadb  # noqa: E402
import config  # noqa: E402  (engine/config.py)
import explain  # noqa: E402
import highlight  # noqa: E402
import retriever  # noqa: E402
from samples import SAMPLES  # noqa: E402

st.set_page_config(page_title="Kat Intelligence - Asteron Global",
                   page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- keys
def get_api_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    for candidate in (APP_DIR / ".streamlit" / "secrets.toml",
                      APP_DIR.parent / "pipeline" / ".streamlit"
                      / "secrets.toml"):
        if candidate.exists():
            try:
                data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("OPENAI_API_KEY"):
                return str(data["OPENAI_API_KEY"]).strip()
    return None


@st.cache_resource
def get_collection():
    try:
        client = chromadb.PersistentClient(path=str(config.chroma_dir()))
        return client.get_collection(config.COLLECTION_NAME)
    except Exception:
        return None


def logo_path() -> Path | None:
    for candidate in (APP_DIR / "assets" / "asteron-logo.svg",
                      APP_DIR.parent / "assets" / "asteron-logo.svg"):
        if candidate.exists():
            return candidate
    return None


def load_manifest() -> list[dict]:
    path = config.policies_dir() / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def kb_stats() -> dict:
    """Numbers for the pipeline tab, from whatever is on disk."""
    stats: dict = {}
    packaged = APP_DIR / "data" / "kb_stats.json"
    if packaged.exists():
        stats.update(json.loads(packaged.read_text(encoding="utf-8")))
    extracted = (APP_DIR.parent / "pipeline" / "storage" / "extracted")
    if extracted.exists():
        docs = [json.loads(p.read_text(encoding="utf-8"))
                for p in extracted.glob("*.json")]
        stats.setdefault("documents", len(docs))
        stats.setdefault("pages", sum(d["pages"] for d in docs))
        stats.setdefault("segments", sum(len(d["segments"]) for d in docs))
    chunks_dir = APP_DIR.parent / "pipeline" / "storage" / "chunks"
    if chunks_dir.exists():
        stats.setdefault(
            "chunks",
            sum(1 for p in chunks_dir.glob("*.jsonl")
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()))
    return stats


# ---------------------------------------------------------------- state
for key, default in (("complaint", ""), ("result", None),
                     ("explanation", ""), ("previews", {})):
    if key not in st.session_state:
        st.session_state[key] = default

collection = get_collection()
api_key = get_api_key()

# ---------------------------------------------------------------- header
logo = logo_path()
cols = st.columns([1, 5])
with cols[0]:
    if logo:
        st.image(str(logo))
with cols[1]:
    st.title("Kat Intelligence")
    st.caption("A GuardKat proof of concept · Demonstration tenant: "
               "Asteron Global (fictional) · Every policy below is fake "
               "and written for this demo.")

# ---------------------------------------------------------------- sidebar
MODES = {"Threshold (recommended)": "threshold",
         "Fixed Top-K": "fixed_k",
         "Dynamic K (stop at the cliff)": "dynamic_k"}
with st.sidebar:
    st.header("Retrieval controls")
    mode_label = st.radio("Mode", list(MODES), index=0)
    mode = MODES[mode_label]
    threshold_pct = st.slider(
        "Relevance threshold (%)", 5, 95,
        int(config.THRESHOLD_DEFAULT * 100),
        disabled=(mode == "fixed_k"),
        help="Chunks scoring below this are not used. In threshold and "
             "dynamic modes, a complaint can honestly match nothing.")
    top_k = st.number_input("Top-K (fixed mode)", 1, 10,
                            config.TOP_K_DEFAULT,
                            disabled=(mode != "fixed_k"))
    max_k = st.number_input("Maximum K (threshold / dynamic)", 1, 10,
                            config.MAX_K_DEFAULT,
                            disabled=(mode == "fixed_k"))
    st.divider()
    if collection is not None:
        st.metric("Policy chunks in the knowledge base",
                  collection.count())
    else:
        st.warning("Knowledge base not built yet.")
    st.write("OpenAI key:", "loaded" if api_key else "not found")
    st.caption("Built with the pipeline in this repository: page-aware "
               "extraction, chunking, text-embedding-3-small, ChromaDB, "
               "gpt-4o-mini explanations grounded only in the chunks "
               "shown.")


def run_retrieval(complaint: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def embed_fn(text: str):
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL,
                                        input=[text])
        return resp.data[0].embedding

    result = retriever.retrieve(
        collection, embed_fn, complaint, mode,
        top_k=int(top_k), threshold=threshold_pct / 100.0,
        max_k=int(max_k))
    return client, result


def stream_explanation(client, complaint: str, used: list[dict]) -> str:
    stream = client.chat.completions.create(
        model=config.CHAT_MODEL, temperature=0.2, stream=True,
        messages=explain.build_messages(complaint, used))
    return st.write_stream(
        chunk.choices[0].delta.content or "" for chunk in stream)


def show_matches(used: list[dict]):
    for i, hit in enumerate(used, start=1):
        pct = round(hit["similarity"] * 100, 1)
        st.markdown(
            f"**{pct}% match** · {hit['title']} · {hit['section']} · "
            f"page {hit['page_start']}")
        with st.expander("Read the passage"):
            st.write(hit["text"])
        preview_key = hit["chunk_id"]
        if st.button(f"Preview page {hit['page_start']} with the passage "
                     f"highlighted", key=f"preview_{preview_key}"):
            with st.spinner("Rendering the page..."):
                st.session_state.previews[preview_key] = \
                    highlight.render_for_hit(hit)
        if preview_key in st.session_state.previews:
            rendered = st.session_state.previews[preview_key]
            st.image(rendered["png"])
            note = (f"Page {rendered['page']} of "
                    f"{rendered['pages_total']} in {hit['title']}")
            note += (" — passage highlighted." if rendered["highlighted"]
                     else " — the passage could not be highlighted "
                          "automatically; the page is shown as-is.")
            st.caption(note)
        st.divider()


tab_home, tab_policies, tab_pipeline, tab_lab = st.tabs(
    ["🏠 Playground", "📄 Policies", "⚙️ How it was built",
     "🔬 Retrieval Lab"])

# ---------------------------------------------------------------- home
with tab_home:
    st.subheader("Imagine you work at Asteron Global. What happened?")
    st.write("Describe a workplace problem in your own words. Kat "
             "Intelligence searches Asteron's policy documents and "
             "shows the policies that match, with the exact page for "
             "each one. Try a sample, or write your own.")
    chip_cols = st.columns(3)
    for i, sample in enumerate(SAMPLES):
        if chip_cols[i % 3].button(sample["label"],
                                   key=f"sample_{i}"):
            st.session_state.complaint = sample["complaint"]
            st.session_state.run_now = True
    complaint = st.text_area(
        "Your complaint", value=st.session_state.complaint, height=130,
        placeholder="e.g. My manager asked me to...")
    run = st.button("Find matching policies", type="primary")
    if run:
        st.session_state.complaint = complaint
        st.session_state.run_now = True

    if st.session_state.get("run_now"):
        st.session_state.run_now = False
        st.session_state.previews = {}
        text = st.session_state.complaint.strip()
        if not text:
            st.warning("Write a complaint first, or pick a sample above.")
        elif collection is None:
            st.error("The knowledge base is not built yet.")
        elif not api_key:
            st.error("No OpenAI key is configured, so the complaint "
                     "cannot be embedded.")
        else:
            try:
                with st.spinner("Searching Asteron's policies..."):
                    client, result = run_retrieval(text)
                st.session_state.result = result
                st.session_state.explanation = ""
                if result["grounded"]:
                    st.session_state.explanation = stream_explanation(
                        client, text, result["used"])
                else:
                    st.session_state.result = result
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.session_state.result = None

    result = st.session_state.result
    if result is not None:
        if result["grounded"]:
            used = result["used"]
            top_pct = round(used[0]["similarity"] * 100, 1)
            st.success(f"Top match {top_pct}% · {len(used)} policy "
                       f"passage(s) used · mode: {result['mode']}")
            if st.session_state.explanation:
                st.markdown(st.session_state.explanation)
            st.subheader("Policies that match")
            show_matches(used)
        else:
            st.warning(config.NO_MATCH_MESSAGE)
            if result["candidates"]:
                with st.expander("Why nothing matched — the closest "
                                 "passages were below the threshold"):
                    for hit in result["candidates"][:3]:
                        pct = round(hit["similarity"] * 100, 1)
                        st.write(f"{pct}% · {hit['title']} · "
                                 f"{hit['section']} · page "
                                 f"{hit['page_start']}")

# ---------------------------------------------------------------- policies
with tab_policies:
    st.subheader("Asteron Global policy library")
    st.write("The ten fictional policies the knowledge base was built "
             "from. In the real GuardKat feature, these are the tenant's "
             "own documents, visible only inside that tenant.")
    manifest = load_manifest()
    if not manifest:
        st.info("The policy manifest is not packaged with this app yet.")
    for entry in manifest:
        pdf_path = config.policies_dir() / entry["file"]
        cols = st.columns([4, 2, 2, 1])
        cols[0].write(f"**{entry['title']}**")
        cols[1].write(entry["owner"])
        cols[2].write(entry["version"])
        if pdf_path.exists():
            cols[3].download_button(
                "PDF", pdf_path.read_bytes(), file_name=entry["file"],
                mime="application/pdf", key=f"dl_{entry['slug']}")

# ---------------------------------------------------------------- pipeline
with tab_pipeline:
    st.subheader("How the knowledge base was built")
    st.write("Documents → page-aware extraction → cleaning → chunking → "
             "embeddings → ChromaDB. The same pipeline shape as a "
             "production RAG system, small enough to inspect end to end.")
    stats = kb_stats()
    if stats:
        cols = st.columns(4)
        cols[0].metric("Policy documents", stats.get("documents", "—"))
        cols[1].metric("Pages read", stats.get("pages", "—"))
        cols[2].metric("Chunks created", stats.get("chunks", "—"))
        cols[3].metric("Embedding cost",
                       f"${stats['cost_usd']:.4f}"
                       if "cost_usd" in stats else "—")
    st.markdown(
        "**Extraction.** Each PDF is read page by page with PyMuPDF. "
        "The page number and the section heading are captured for every "
        "passage and carried through the whole pipeline, which is what "
        "makes the exact-page preview possible. No LLM is used here.\n\n"
        "**Cleaning.** Conservative whitespace cleanup only; the words "
        "of the policy are never rewritten.\n\n"
        "**Chunking.** Structure first: a chunk never crosses a section "
        "boundary, is capped at 1,600 characters, and carries a small "
        "overlap from the previous chunk.\n\n"
        "**Embeddings.** text-embedding-3-small turns each chunk into a "
        "1,536-number vector. A hash of each chunk's text is stored, so "
        "re-running the pipeline never pays to embed the same text "
        "twice.\n\n"
        "**The store.** ChromaDB is a folder of files that ships with "
        "this app. It is enough for a proof of concept; the production "
        "design swaps it for Azure AI Search, and the chunks and "
        "embeddings move across unchanged.")

# ---------------------------------------------------------------- lab
with tab_lab:
    st.subheader("Retrieval, augmentation, generation — in the open")
    st.write("The three stages of RAG, shown separately. Retrieval "
             "finds passages, augmentation assembles them into the "
             "prompt, generation writes the answer from that prompt "
             "only. Uses the sidebar controls.")
    lab_text = st.text_area("Complaint to inspect",
                            value=SAMPLES[0]["complaint"], height=110,
                            key="lab_complaint")
    if st.button("Run the stages", key="lab_run"):
        if collection is None or not api_key:
            st.error("Needs the knowledge base and an OpenAI key.")
        else:
            try:
                client, result = run_retrieval(lab_text)
                st.markdown("**Stage 1 · Retrieval** — every candidate, "
                            "best first. ✓ = used after the mode filter.")
                used_ids = {h["chunk_id"] for h in result["used"]}
                rows = [{
                    "Used": "✓" if h["chunk_id"] in used_ids else "—",
                    "Match %": round(h["similarity"] * 100, 1),
                    "Document": h["title"], "Section": h["section"],
                    "Page": h["page_start"],
                } for h in result["candidates"]]
                st.dataframe(rows, use_container_width=True)
                st.markdown("**Stage 2 · Augmentation** — the exact "
                            "prompt the model receives.")
                if result["used"]:
                    messages = explain.build_messages(lab_text,
                                                      result["used"])
                    st.code(messages[0]["content"] + "\n\n---\n\n"
                            + messages[1]["content"])
                    st.markdown("**Stage 3 · Generation** — the answer "
                                "written from that prompt.")
                    answer = explain.explain(client, lab_text,
                                             result["used"])
                    st.write(answer)
                else:
                    st.info("No passages passed the filter, so no prompt "
                            "is built and the model is never called. "
                            "That restraint is deliberate.")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
