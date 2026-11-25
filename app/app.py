"""Kat Intelligence - Asteron Global playground (the public app).

Home is the playground: describe a workplace complaint as if you were
an Asteron employee, and Kat Intelligence finds the company policies
that match it, explains why, and opens the policy PDF at the exact
page, with the passage highlighted.

Tabs: Playground - Policies - How it was built - Retrieval Lab.
"""

import base64
import io
import json
import os
import sys
import tomllib
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ENGINE_DIR = APP_DIR / "engine"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(APP_DIR))

import chromadb  # noqa: E402
import config  # noqa: E402  (engine/config.py)
import explain  # noqa: E402
import highlight  # noqa: E402
import retriever  # noqa: E402
from samples import SAMPLES  # noqa: E402

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


def count_tokens(text: str) -> int | None:
    """Token count for cost display; None when tiktoken is absent."""
    return len(_enc.encode(text)) if _enc is not None else None

def _page_icon():
    for candidate in (APP_DIR / "assets" / "guardkat-logo.png",
                      APP_DIR.parent / "assets" / "guardkat-logo.png"):
        if candidate.exists():
            return str(candidate)
    try:
        from logo_data import GUARDKAT_LOGO_PNG_B64
        from PIL import Image
        return Image.open(io.BytesIO(
            base64.b64decode(GUARDKAT_LOGO_PNG_B64)))
    except Exception:
        return "🛡️"


st.set_page_config(page_title="Kat Intelligence · by GuardKat",
                   page_icon=_page_icon(), layout="wide")


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


def asset_path(name: str) -> Path | None:
    for candidate in (APP_DIR / "assets" / name,
                      APP_DIR.parent / "assets" / name):
        if candidate.exists():
            return candidate
    return None


def img_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix == ".png" else "image/svg+xml"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def guardkat_logo_bytes() -> bytes | None:
    """The GuardKat logo: the real PNG file when it is present,
    otherwise the copy embedded in logo_data.py. The PNG is a binary
    picture and the GitHub route for this repo carries text only, so
    the picture rides inside the code as base64 text."""
    path = asset_path("guardkat-logo.png")
    if path:
        return path.read_bytes()
    try:
        from logo_data import GUARDKAT_LOGO_PNG_B64
        return base64.b64decode(GUARDKAT_LOGO_PNG_B64)
    except Exception:
        return None


def guardkat_logo_uri() -> str | None:
    raw = guardkat_logo_bytes()
    if raw is None:
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def load_manifest() -> list[dict]:
    path = config.policies_dir() / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def pipeline_settings() -> dict:
    """The decided pipeline configuration, read live from
    pipeline/config.py when it is on disk, so the web view can never
    drift from the code that actually ran."""
    settings: dict = {}
    try:
        import importlib.util
        path = APP_DIR.parent / "pipeline" / "config.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "pipeline_config", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            settings = {
                "chunk_max": mod.CHUNK_MAX_CHARS,
                "chunk_overlap": mod.CHUNK_OVERLAP_CHARS,
                "chunk_min_tail": mod.CHUNK_MIN_TAIL_CHARS,
                "min_cleaned": mod.MIN_CLEANED_CHARS,
                "embedding_model": mod.EMBEDDING_MODEL,
                "embedding_dims": mod.EMBEDDING_DIMENSIONS,
                "embedding_price":
                    mod.EMBEDDING_PRICE_USD_PER_1M_DEFAULT,
                "collection": mod.COLLECTION_NAME,
            }
    except Exception:
        pass
    return settings


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
        chunk_texts = []
        for p in chunks_dir.glob("*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunk_texts.append(json.loads(line)["text"])
        stats.setdefault("chunks", len(chunk_texts))
        if _enc is not None:
            stats.setdefault(
                "tokens",
                sum(len(_enc.encode(t)) for t in chunk_texts))
    return stats


# ---------------------------------------------------------------- state
for key, default in (("complaint", ""), ("result", None),
                     ("explanation", ""), ("previews", {}),
                     ("search_cost", None)):
    if key not in st.session_state:
        st.session_state[key] = default

collection = get_collection()
api_key = get_api_key()

# ---------------------------------------------------------------- header
BRAND_CSS = """
<style>
.gk-brand {text-align: center; padding: 0.4rem 0 0.2rem;}
.gk-logos {display: flex; justify-content: center; align-items: center;
  gap: 20px;}
.gk-logo-name {font-size: 11px; letter-spacing: 2px; color: #607d8b;
  font-weight: 600; margin-top: 4px;}
.gk-x {font-size: 22px; color: #90a4ae;}
.gk-title {font-size: 2.4rem; margin: 0.6rem 0 0.1rem; color: #263238;}
.gk-sub {color: #546e7a; font-size: 0.98rem; margin: 0 auto;
  max-width: 660px;}
.gk-loader {display: flex; flex-direction: column; align-items: center;
  padding: 16px 0 4px;}
.gk-ring {position: relative; width: 104px; height: 104px;}
.gk-ring:before {content: ""; position: absolute; inset: 0;
  border-radius: 50%; border: 5px solid #e0e0e0;
  border-top-color: #d98e2b; animation: gkspin 1.4s linear infinite;}
.gk-ring img {position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%); height: 62px;}
@keyframes gkspin {to {transform: rotate(360deg);}}
.gk-loader-text {margin-top: 10px; color: #546e7a; font-size: 0.92rem;}
.gk-footer {text-align: center; color: #78909c; font-size: 0.82rem;}
</style>
"""


def render_brand_header():
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    blocks = []
    gk_uri = guardkat_logo_uri()
    if gk_uri:
        blocks.append(
            f'<div><img src="{gk_uri}" style="height:74px">'
            '<div class="gk-logo-name">GUARDKAT</div></div>')
    ast = asset_path("asteron-logo.svg")
    if ast:
        # No written caption here: the Asteron logo already carries
        # the words ASTERON GLOBAL inside the image itself.
        blocks.append(
            f'<div><img src="{img_data_uri(ast)}" style="height:74px">'
            '</div>')
    logos = '<div class="gk-x">×</div>'.join(blocks)
    st.markdown(
        f'<div class="gk-brand"><div class="gk-logos">{logos}</div>'
        '<h1 class="gk-title">Kat Intelligence</h1>'
        '<p class="gk-sub">GuardKat\u2019s document intelligence, '
        'running on a fictional company. When a concern is reported, '
        'it finds the policies that apply and opens them on the exact '
        'page.</p></div>',
        unsafe_allow_html=True)


render_brand_header()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    _gk_bytes = guardkat_logo_bytes()
    if _gk_bytes:
        _brand_cols = st.columns([1, 3])
        _brand_cols[0].image(_gk_bytes, width=44)
        _brand_cols[1].markdown("**GuardKat**\n\nKat Intelligence")
    st.header("Retrieval controls")
    st.write("Two independent filters. The threshold decides which "
             "passages are relevant enough. K decides how many of "
             "those passages are used. Both are always on, and "
             "neither switches the other off.")
    threshold_pct = st.slider(
        "Relevance threshold (%)", 5, 95,
        int(config.THRESHOLD_DEFAULT * 100),
        help="Passages scoring below this are never used, whatever K "
             "says. If nothing clears it, the complaint honestly "
             "matches nothing.")
    k_label = st.radio(
        "How many passages (K)",
        ["Dynamic — stop at the cliff", "Fixed number"], index=0,
        help="Dynamic keeps taking passages while their scores stay "
             "close to the previous one, and stops where they fall "
             "away. Fixed takes exactly the fixed number below.")
    k_mode = "dynamic" if k_label.startswith("Dynamic") else "fixed"
    fixed_k = st.number_input("Passages to use (fixed K)", 1, 10,
                              config.TOP_K_DEFAULT)
    max_k = st.number_input("Maximum passages (dynamic K cap)", 1, 10,
                            config.MAX_K_DEFAULT)
    st.divider()
    if collection is not None:
        st.metric("Policy chunks in the knowledge base",
                  collection.count())
    else:
        st.warning("Knowledge base not built yet.")
    st.write("OpenAI key:", "loaded" if api_key else "not found")
    st.divider()
    st.markdown("**Models & cost**")
    st.write(f"Embedding model: {config.EMBEDDING_MODEL}")
    st.write(f"Explanation model: {config.CHAT_MODEL}")
    price = st.number_input(
        "Embedding price (USD per 1M tokens)", min_value=0.0,
        value=float(os.environ.get("EMBEDDING_PRICE_USD", 0.02)),
        step=0.01, format="%.4f", key="embedding_price",
        help="The price is an input, never a hidden constant. It "
             "drives every cost figure in this app.")
    st.caption("Built with the pipeline in this repository: page-aware "
               "extraction, chunking, text-embedding-3-small, ChromaDB, "
               "gpt-4o-mini explanations grounded only in the chunks "
               "shown.")


def show_loader(slot, text: str):
    """The GuardKat loader: the meerkat inside a spinning ring, the
    same pattern as the GuardKat website's loading screen."""
    gk_uri = guardkat_logo_uri()
    img = f'<img src="{gk_uri}">' if gk_uri else ""
    slot.markdown(
        f'<div class="gk-loader"><div class="gk-ring">{img}</div>'
        f'<div class="gk-loader-text">{text}</div></div>',
        unsafe_allow_html=True)


def run_retrieval(complaint: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def embed_fn(text: str):
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL,
                                        input=[text])
        return resp.data[0].embedding

    result = retriever.retrieve(
        collection, embed_fn, complaint,
        threshold=threshold_pct / 100.0, k_mode=k_mode,
        fixed_k=int(fixed_k), max_k=int(max_k))
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
            slot = st.empty()
            show_loader(slot, "Opening the policy at the exact page…")
            st.session_state.previews[preview_key] = \
                highlight.render_for_hit(hit)
            slot.empty()
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
    st.markdown("### What is this?")
    st.write(
        "Every company has policies: a code of conduct, rules on gifts, "
        "harassment, expenses, data and more. The problem is finding the "
        "right one at the moment it matters. When an employee reports a "
        "concern, the person handling the case usually has to search "
        "shared drives and old emails to find which policy applies.\n\n"
        "Kat Intelligence removes that search. The company's own policy "
        "documents are read in advance and kept ready. The moment a "
        "concern is reported, it finds the passages that apply, explains "
        "them in plain words, and opens each document on the exact page.")
    st.markdown("### Why companies use it")
    benefit_cols = st.columns(3)
    with benefit_cols[0]:
        with st.container(border=True):
            st.markdown("**The right policy, in seconds**")
            st.write("The matching policies arrive with the case. "
                     "Nobody digs through folders, and a new investigator "
                     "is as well briefed as a senior one.")
    with benefit_cols[1]:
        with st.container(border=True):
            st.markdown("**The exact page, highlighted**")
            st.write("Every match opens the real document on the page "
                     "that matters, with the passage highlighted, so "
                     "anyone can check it with one click.")
    with benefit_cols[2]:
        with st.container(border=True):
            st.markdown("**Honest when nothing matches**")
            st.write("If no policy covers the concern, it says so "
                     "instead of guessing. People make the decisions; "
                     "Kat Intelligence brings the right pages.")
    st.markdown("### Where it helps")
    use_cols = st.columns(4)
    use_cases = [
        ("Speak-up reports",
         "A whistleblowing report arrives, and the policies that govern "
         "it are attached to the case from the start."),
        ("Harassment inquiries",
         "The inquiry committee sees the exact clauses, timelines and "
         "steps the policy promises."),
        ("Compliance checks",
         "Gifts, conflicts of interest and expense claims checked "
         "against the written rule, not memory."),
        ("Employee questions",
         "\u201cCan I accept this?\u201d answered from the policy "
         "itself, with the page to prove it."),
    ]
    for col, (title, body) in zip(use_cols, use_cases):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(body)
    st.divider()
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
            slot = st.empty()
            try:
                show_loader(slot, "Kat Intelligence is reading "
                                  "Asteron's policies…")
                client, result = run_retrieval(text)
                slot.empty()
                _tokens = count_tokens(text)
                st.session_state.search_cost = (
                    {"tokens": _tokens,
                     "cost": _tokens / 1_000_000 * price}
                    if _tokens is not None else None)
                st.session_state.result = result
                st.session_state.explanation = ""
                if result["grounded"]:
                    st.session_state.explanation = stream_explanation(
                        client, text, result["used"])
                else:
                    st.session_state.result = result
            except Exception as exc:
                slot.empty()
                st.error(f"Something went wrong: {exc}")
                st.session_state.result = None

    result = st.session_state.result
    if result is not None:
        if st.session_state.search_cost:
            _sc = st.session_state.search_cost
            st.caption(
                f"This search embedded your complaint ({_sc['tokens']} "
                f"tokens, about ${_sc['cost']:.6f} at ${price:.2f} per "
                f"1M tokens) with {config.EMBEDDING_MODEL}. The "
                f"explanation was written by {config.CHAT_MODEL}.")
        if result["grounded"]:
            used = result["used"]
            top_pct = round(used[0]["similarity"] * 100, 1)
            _s = result["settings"]
            _k_text = ("dynamic K" if _s["k_mode"] == "dynamic"
                       else f"fixed K = {_s['fixed_k']}")
            st.success(f"Top match {top_pct}% · {len(used)} policy "
                       f"passage(s) used · threshold "
                       f"{int(_s['threshold'] * 100)}% · {_k_text}")
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
        cols[3].metric(
            "Embedding cost",
            f"${stats['cost_usd']:.4f}" if "cost_usd" in stats
            else f"${stats['tokens'] / 1_000_000 * price:.4f}"
            if "tokens" in stats else "—")
    st.markdown("**The settings we decided** — read live from the "
                "configuration files, so what you see here is exactly "
                "what runs. Nothing is hidden in the code.")
    _p = pipeline_settings()
    _prep = [
        ("Chunk size (max characters)", _p.get("chunk_max", "—")),
        ("Overlap between chunks (characters)",
         _p.get("chunk_overlap", "—")),
        ("Smallest tail kept (characters)",
         _p.get("chunk_min_tail", "—")),
        ("Shortest text kept after cleaning (characters)",
         _p.get("min_cleaned", "—")),
        ("Embedding model",
         _p.get("embedding_model", config.EMBEDDING_MODEL)),
        ("Numbers per chunk (vector size)",
         _p.get("embedding_dims", "—")),
        ("Embedding price (USD per 1M tokens)",
         _p.get("embedding_price", "—")),
        ("Knowledge base collection",
         _p.get("collection", config.COLLECTION_NAME)),
    ]
    _find = [
        ("Default K behaviour", "Dynamic — stop at the cliff"),
        ("Relevance threshold",
         f"{int(config.THRESHOLD_DEFAULT * 100)}%"),
        ("Fixed K (passages)", config.TOP_K_DEFAULT),
        ("Maximum passages (dynamic K cap)", config.MAX_K_DEFAULT),
        ("Dynamic K cliff (points)",
         int(config.DYNAMIC_DROP_POINTS * 100)),
        ("Candidates fetched before filtering", config.CANDIDATE_POOL),
        ("Explanation model", config.CHAT_MODEL),
        ("Explanation temperature", 0.2),
    ]
    _set_cols = st.columns(2)
    with _set_cols[0]:
        st.markdown("*Preparing the documents*")
        st.dataframe([{"Setting": k, "Value": str(v)} for k, v in _prep],
                     hide_index=True, use_container_width=True)
    with _set_cols[1]:
        st.markdown("*Finding the answers*")
        st.dataframe([{"Setting": k, "Value": str(v)} for k, v in _find],
                     hide_index=True, use_container_width=True)
    if "tokens" in stats:
        st.markdown("**What it costs** — embedding is a one-time cost "
                    "per document. Re-runs skip unchanged text, so "
                    "updates pay only for what changed.")
        _cost_cols = st.columns(3)
        _cost_cols[0].metric("Tokens in the knowledge base",
                             stats["tokens"])
        _cost_cols[1].metric("Embedding price (per 1M tokens)",
                             f"${price:.4f}")
        _cost_cols[2].metric(
            "Cost to build this knowledge base",
            f"${stats['tokens'] / 1_000_000 * price:.4f}")
    st.markdown(
        "**1 · Extract — what we did.** PyMuPDF reads each PDF page "
        "by page, in blocks. Every block keeps its page number and "
        "the current section heading, taken from numbered headings; "
        "the title and version lines on page 1 are skipped as "
        "metadata. No LLM is used here. **What we assumed:** the "
        "letterhead and footer sit in fixed bands at the top and "
        "bottom of every page and are dropped by position — true "
        "for documents from one template; a generic handler would "
        "instead detect lines that repeat across pages.\n\n"
        "**2 · Clean — what we did.** Whitespace only: line endings "
        "normalised, runs of spaces collapsed, extra blank lines "
        "removed. **What we assumed:** the policy's own words are "
        "the source of truth, so no word is ever added, removed or "
        "rewritten.\n\n"
        "**3 · Chunk — what we did.** Segments are packed in order "
        "into chunks of at most 1,600 characters, never crossing a "
        "section boundary, with a short overlap carried from the "
        "previous chunk's tail and a too-short final chunk merged "
        "into its neighbour. Each chunk records its document, "
        "section and page range — which is what makes the exact-page "
        "preview possible. **What we assumed:** a section is the "
        "natural unit of meaning in a policy, so chunks hold whole "
        "sections, not arbitrary slices.\n\n"
        "**4 · Embed — what we did.** text-embedding-3-small turns "
        "each chunk into a 1,536-number vector. Tokens are counted "
        "and the cost estimated before any API call. Each chunk's "
        "text is hashed with SHA-256; a chunk whose hash already "
        "has an embedding is skipped, so re-runs never pay twice. "
        "If a policy changes, the changed text produces a new hash "
        "that does not match, so exactly those chunks are embedded "
        "again. **What we assumed:** identical text always deserves "
        "the identical vector.\n\n"
        "**5 · Store — what we did.** Chunks and vectors are "
        "upserted into ChromaDB by chunk id, with title, section "
        "and page range as metadata, so a changed chunk replaces "
        "its old vector instead of duplicating. **What we assumed:** "
        "one collection per tenant is enough at this scale; the "
        "production design swaps the folder for Azure AI Search, "
        "and the chunks and embeddings move across unchanged.")

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
                            "best first. ✓ = used after the threshold "
                            "and K filters.")
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

# ---------------------------------------------------------------- footer
st.divider()
st.markdown(
    '<p class="gk-footer">Kat Intelligence is a GuardKat concept, '
    'shown here as a working proof of concept. Asteron Global is a '
    'fictional company and its policies are samples written for this '
    'demo. In GuardKat, the same idea works on a company\u2019s own '
    'policies, inside its own tenant, visible only to its own '
    'case team.</p>',
    unsafe_allow_html=True)
