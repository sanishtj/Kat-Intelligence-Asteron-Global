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


def exchange_cost_usd(query_tokens, usage, emb_price, in_price,
                      out_price) -> float:
    """The full cost of one exchange: embedding the question, plus
    the chat call when an answer was generated."""
    total = (query_tokens or 0) / 1_000_000 * emb_price
    if usage:
        total += usage.get("prompt_tokens", 0) / 1_000_000 * in_price
        total += (usage.get("completion_tokens", 0)
                  / 1_000_000 * out_price)
    return total


@st.cache_data(show_spinner=False)
def kb_embedding_tokens(_collection, count: int):
    """Total tokens embedded to build this knowledge base, counted
    from the stored chunk texts with the same tokenizer the
    embedder used, so the burn figure is real wherever the store
    can be read."""
    if _enc is None:
        return None
    try:
        documents = _collection.get()["documents"]
        return sum(len(_enc.encode(doc)) for doc in documents)
    except Exception:
        return None


def chat_cost_caption(tokens, emb_price, in_price, out_price):
    """One quiet caption line under a chat answer: what the
    exchange used and what it cost."""
    if not tokens:
        return
    q = tokens.get("query", 0)
    p = tokens.get("prompt", 0)
    c = tokens.get("completion", 0)
    cost = exchange_cost_usd(
        q, {"prompt_tokens": p, "completion_tokens": c},
        emb_price, in_price, out_price)
    if p or c:
        st.caption(f"Search and answer: {q + p + c:,} tokens · "
                   f"about ${cost:.6f}")
    else:
        st.caption(f"Policy search: {q:,} tokens · "
                   f"about ${cost:.6f}")

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
                     ("search_cost", None), ("policy_preview", None),
                     ("chat_messages", []), ("chat_pending", None)):
    if key not in st.session_state:
        st.session_state[key] = default

collection = get_collection()
api_key = get_api_key()

# ---------------------------------------------------------------- header
BRAND_CSS = """
<style>
/* Sidebar chrome: Streamlit leaves a tall empty gap above the
   sidebar content to clear its own collapse button, and the button
   itself renders near-white on white. Trim the gap so the brand
   sits near the top, and make the collapse control properly
   visible (never hidden: it must stay usable). */
[data-testid="stSidebarUserContent"] {padding-top: 0.25rem;}
[data-testid="stSidebarCollapseButton"] {opacity: 1; color: #263238;
  background: rgba(38, 50, 56, 0.06); border-radius: 6px;}
[data-testid="stSidebarCollapsedControl"] {opacity: 1; color: #263238;
  background: rgba(38, 50, 56, 0.06); border-radius: 6px;}
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
/* Top spacing: Streamlit reserves a tall band at the top of the
   main area and above the sidebar header. Pull the content up so
   both brands sit near the top where they belong. */
[data-testid="stMainBlockContainer"] {padding-top: 1.6rem;}
[data-testid="stAppViewBlockContainer"] {padding-top: 1.6rem;}
[data-testid="stSidebarHeader"] {padding-top: 0.4rem;
  padding-bottom: 0;}
/* Tabs that read as tabs: larger labels, boxed tops, and the
   selected tab white with the brand amber line under it. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 6px;}
[data-testid="stTabs"] [data-baseweb="tab"] {font-size: 1.05rem;
  font-weight: 600; color: #546e7a; padding: 10px 18px;
  background: rgba(38, 50, 56, 0.05);
  border: 1px solid rgba(38, 50, 56, 0.16); border-bottom: none;
  border-radius: 10px 10px 0 0;}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]
  {background: #ffffff; color: #263238;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]
  {background-color: #d98e2b; height: 3px;}
/* The chat box is the centrepiece of the Chat tab: a framed,
   tinted panel with a large input, not a bare text box. */
[data-testid="stForm"] {border: 2px solid #d98e2b;
  border-radius: 14px; background: #fffaf1;
  padding: 14px 16px 6px;
  box-shadow: 0 2px 12px rgba(38, 50, 56, 0.10);}
[data-testid="stForm"] [data-testid="stTextInput"] input
  {font-size: 1.1rem; padding: 0.65rem 0.8rem;}
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
        'running on a fictional company.<br>'
        'When a concern is reported, '
        'it finds the policies that apply and opens them on the exact '
        'page.</p></div>',
        unsafe_allow_html=True)


render_brand_header()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    _gk_uri = guardkat_logo_uri()
    _brand_img = (f'<img src="{_gk_uri}" style="height:64px">'
                  if _gk_uri else "")
    st.markdown(
        '<div style="text-align:center; padding:6px 0 2px 0">'
        f'{_brand_img}'
        '<div style="font-weight:700; font-size:17px; color:#263238; '
        'margin-top:6px">GuardKat</div>'
        '<div style="font-size:12px; letter-spacing:1.5px; '
        'color:#607d8b">KAT INTELLIGENCE</div></div>',
        unsafe_allow_html=True)
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
        help="Dynamic takes the passages above the threshold in "
             "score order while their scores stay close together. It "
             "stops at a cliff: a sudden drop of more than 8 points "
             "between one passage and the next. It also stops at the "
             "maximum below, whichever comes first. Example: scores "
             "62, 58 and 41 are all above a 35 threshold, but only "
             "62 and 58 are used, because 41 drops off a cliff. "
             "Fixed uses exactly the fixed number below.")
    k_mode = "dynamic" if k_label.startswith("Dynamic") else "fixed"
    fixed_k = st.number_input(
        "Passages to use (fixed K)", 1, 10, config.TOP_K_DEFAULT,
        disabled=(k_mode == "dynamic"),
        help="Exactly this many of the passages above the threshold "
             "are used, best scores first. Enabled when Fixed number "
             "is selected above.")
    max_k = st.number_input(
        "Maximum passages (dynamic K cap)", 1, 10,
        config.MAX_K_DEFAULT, disabled=(k_mode == "fixed"),
        help="The most passages dynamic can use. It may use fewer: "
             "it stops early at a cliff, a sudden drop of more than "
             "8 points between one passage and the next. Enabled "
             "when Dynamic is selected above.")
    st.divider()
    if collection is not None:
        st.metric("Policy chunks in the knowledge base",
                  collection.count())
        _kb_tokens = kb_embedding_tokens(collection,
                                         collection.count())
        if _kb_tokens:
            _emb_price_now = float(st.session_state.get(
                "embedding_price",
                float(os.environ.get("EMBEDDING_PRICE_USD", 0.02))))
            st.caption(
                f"Embedding this knowledge base: {_kb_tokens:,} "
                f"tokens · about "
                f"${_kb_tokens / 1_000_000 * _emb_price_now:.4f} "
                f"at the embedding price below")
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
    chat_in_price = st.number_input(
        "Chat model input price (USD per 1M tokens)", min_value=0.0,
        value=0.15, step=0.01, format="%.4f", key="chat_input_price",
        help="What the explanation model charges for the text it "
             "reads: the question plus the policy passages sent "
             "with it. The gpt-4o-mini list price is the starting "
             "value; change it any time.")
    chat_out_price = st.number_input(
        "Chat model output price (USD per 1M tokens)", min_value=0.0,
        value=0.60, step=0.01, format="%.4f", key="chat_output_price",
        help="What the explanation model charges for the answer it "
             "writes. The gpt-4o-mini list price is the starting "
             "value.")
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
    captured: dict = {}

    def embed_fn(text: str):
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL,
                                        input=[text])
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", None)
        captured["query_tokens"] = (tokens if tokens is not None
                                    else count_tokens(text))
        return resp.data[0].embedding

    result = retriever.retrieve(
        collection, embed_fn, complaint,
        threshold=threshold_pct / 100.0, k_mode=k_mode,
        fixed_k=int(fixed_k), max_k=int(max_k))
    return client, result, captured.get("query_tokens")


def stream_explanation(client, complaint: str, used: list[dict]):
    """Stream the explanation and return (text, usage). The usage
    rides on the stream's final chunk (include_usage), so every
    answer can carry its real token cost."""
    stream = client.chat.completions.create(
        model=config.CHAT_MODEL, temperature=0.2, stream=True,
        stream_options={"include_usage": True},
        messages=explain.build_messages(complaint, used))
    usage: dict = {}

    def _pieces():
        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage.update({
                    "prompt_tokens": chunk_usage.prompt_tokens,
                    "completion_tokens": chunk_usage.completion_tokens,
                    "total_tokens": chunk_usage.total_tokens})
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""

    return st.write_stream(_pieces()), usage


@st.dialog("The policy page, with the passage highlighted",
           width="large")
def _page_dialog(hit: dict, rendered: dict):
    st.markdown(f"**{hit['title']}** · {hit['section']} · "
                f"page {hit['page_start']}")
    st.image(rendered["png"])
    note = (f"Page {rendered['page']} of {rendered['pages_total']} "
            f"in {hit['title']}")
    note += (" — passage highlighted." if rendered["highlighted"]
             else " — the passage could not be highlighted "
                  "automatically; the page is shown as-is.")
    st.caption(note)
    if st.button("Close"):
        st.rerun()


@st.dialog("Policy preview", width="large")
def _policy_dialog(entry: dict, pdf_bytes: bytes):
    st.markdown(f"**{entry['title']}**")
    # Built-in Streamlit PDF viewer: read and page through the
    # document right here in the pop-up, no download.
    st.pdf(pdf_bytes)
    if st.button("Close"):
        st.rerun()


def show_matches(used: list[dict], key_prefix: str = ""):
    for i, hit in enumerate(used, start=1):
        pct = round(hit["similarity"] * 100, 1)
        st.markdown(
            f"**{pct}% match** · {hit['title']} · {hit['section']} · "
            f"page {hit['page_start']}")
        with st.expander("Read the passage"):
            st.write(hit["text"])
        preview_key = f"{key_prefix}{hit['chunk_id']}"
        if st.button(f"Preview page {hit['page_start']} with the passage "
                     f"highlighted", key=f"preview_{preview_key}"):
            rendered = st.session_state.previews.get(preview_key)
            if rendered is None:
                with st.spinner("Opening the policy at the exact page…"):
                    rendered = highlight.render_for_hit(hit)
                st.session_state.previews[preview_key] = rendered
            _page_dialog(hit, rendered)
        st.divider()


tab_chat, tab_home, tab_policies, tab_pipeline, tab_lab = st.tabs(
    ["💬 Chat", "📋 Playground", "📄 Policies", "⚙️ How it was built",
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
            slot = st.empty()
            try:
                show_loader(slot, "Kat Intelligence is reading "
                                  "Asteron's policies…")
                client, result, query_tokens = run_retrieval(text)
                slot.empty()
                st.session_state.result = result
                st.session_state.explanation = ""
                usage = {}
                if result["grounded"]:
                    st.session_state.explanation, usage = \
                        stream_explanation(client, text, result["used"])
                _chat_tokens = usage.get(
                    "total_tokens",
                    usage.get("prompt_tokens", 0)
                    + usage.get("completion_tokens", 0)) if usage else 0
                st.session_state.search_cost = {
                    "tokens": (query_tokens or 0) + _chat_tokens,
                    "cost": exchange_cost_usd(
                        query_tokens, usage, price, chat_in_price,
                        chat_out_price),
                    "answered": bool(usage)}
            except Exception as exc:
                slot.empty()
                st.error(f"Something went wrong: {exc}")
                st.session_state.result = None

    result = st.session_state.result
    if result is not None:
        if st.session_state.search_cost:
            _sc = st.session_state.search_cost
            st.caption(
                f"This search used {_sc['tokens']:,} tokens and "
                f"cost about ${_sc['cost']:.6f}: the complaint "
                f"embedded with {config.EMBEDDING_MODEL}"
                + (f" and the explanation written by "
                   f"{config.CHAT_MODEL}."
                   if _sc["answered"] else
                   ", and no explanation was generated."))
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

# ---------------------------------------------------------------- chat
with tab_chat:
    chat_col, rail_col = st.columns([3, 1])
    with rail_col:
        with st.expander("What is this? About Kat Intelligence",
                         expanded=False):
            st.markdown("### What is this?")
            st.write(
                "Every company has policies: a code of conduct, rules "
                "on gifts, harassment, expenses, data and more. The "
                "problem is finding the right one at the moment it "
                "matters. When an employee reports a concern, the "
                "person handling the case usually has to search shared "
                "drives and old emails to find which policy applies."
                "\n\n"
                "Kat Intelligence removes that search. The company's "
                "own policy documents are read in advance and kept "
                "ready. The moment a concern is reported, it finds the "
                "passages that apply, explains them in plain words, "
                "and opens each document on the exact page.")
            st.markdown("### Why companies use it")
            with st.container(border=True):
                st.markdown("**The right policy, in seconds**")
                st.write("The matching policies arrive with the case. "
                         "Nobody digs through folders, and a new "
                         "investigator is as well briefed as a senior "
                         "one.")
            with st.container(border=True):
                st.markdown("**The exact page, highlighted**")
                st.write("Every match opens the real document on the "
                         "page that matters, with the passage "
                         "highlighted, so anyone can check it with one "
                         "click.")
            with st.container(border=True):
                st.markdown("**Honest when nothing matches**")
                st.write("If no policy covers the concern, it says so "
                         "instead of guessing. People make the "
                         "decisions; Kat Intelligence brings the right "
                         "pages.")
            st.markdown("### Where it helps")
            use_cases = [
                ("Speak-up reports",
                 "A whistleblowing report arrives, and the policies "
                 "that govern it are attached to the case from the "
                 "start."),
                ("Harassment inquiries",
                 "The inquiry committee sees the exact clauses, "
                 "timelines and steps the policy promises."),
                ("Compliance checks",
                 "Gifts, conflicts of interest and expense claims "
                 "checked against the written rule, not memory."),
                ("Employee questions",
                 "\u201cCan I accept this?\u201d answered from the "
                 "policy itself, with the page to prove it."),
            ]
            for title, body in use_cases:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(body)
        with st.expander("Sample situations", expanded=False):
            st.write("Start from a sample situation:")
            for i, sample in enumerate(SAMPLES):
                if st.button(sample["label"], key=f"chat_chip_{i}",
                             use_container_width=True):
                    st.session_state.chat_pending = sample["complaint"]
    with chat_col:
        st.markdown("### \U0001f4ac Ask Kat Intelligence")
        if collection is None:
            st.warning("The knowledge base is not built yet, so the "
                       "chat cannot look anything up. Build it with "
                       "the pipeline steps first (see the How it was "
                       "built tab).")
        with st.form("chat_form", clear_on_submit=True):
            ask_cols = st.columns([5, 1])
            question = ask_cols[0].text_input(
                "Your question",
                placeholder="Ask about an Asteron policy, or "
                            "describe a situation\u2026",
                label_visibility="collapsed")
            sent = ask_cols[1].form_submit_button(
                "Ask", type="primary", use_container_width=True)
        st.caption("Answers come from the policies themselves, with "
                   "the sources under each answer. When no policy "
                   "covers it, the chat says so instead of guessing.")
        new_text = (question if sent else None) or \
            st.session_state.chat_pending
        if new_text:
            st.session_state.chat_pending = None
            st.session_state.chat_messages.append(
                {"role": "user", "content": new_text})
            assistant_idx = len(st.session_state.chat_messages)
            with st.chat_message("user"):
                st.markdown(new_text)
            tokens = None
            with st.chat_message("assistant"):
                if collection is None:
                    answer = ("The knowledge base is not built yet, "
                              "so I cannot look that up. Build it "
                              "with the pipeline steps first.")
                    used = []
                    st.markdown(answer)
                elif not api_key:
                    answer = ("No OpenAI API key is configured, so I "
                              "cannot search the policies. Add the "
                              "key to the app's secrets and try "
                              "again.")
                    used = []
                    st.markdown(answer)
                else:
                    slot = st.empty()
                    show_loader(slot, "Reading the policies\u2026")
                    client, result, query_tokens = \
                        run_retrieval(new_text)
                    slot.empty()
                    used = result["used"]
                    tokens = {"query": query_tokens or 0,
                              "prompt": 0, "completion": 0}
                    if used:
                        answer, usage = stream_explanation(
                            client, new_text, used)
                        tokens["prompt"] = usage.get(
                            "prompt_tokens", 0)
                        tokens["completion"] = usage.get(
                            "completion_tokens", 0)
                        st.markdown("**Sources**")
                        show_matches(
                            used,
                            key_prefix=f"chat{assistant_idx}_")
                    else:
                        answer = config.NO_MATCH_MESSAGE
                        st.markdown(answer)
                    chat_cost_caption(tokens, price, chat_in_price,
                                      chat_out_price)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer, "used": used,
                 "tokens": tokens})
            st.rerun()
        if st.session_state.chat_messages:
            if st.button("Clear conversation", key="clear_chat"):
                st.session_state.chat_messages = []
                st.session_state.chat_pending = None
                st.rerun()
        messages = st.session_state.chat_messages
        pairs = [(i, messages[i], messages[i + 1])
                 for i in range(0, len(messages) - 1, 2)]
        for user_i, user_msg, asst_msg in reversed(pairs):
            with st.chat_message("user"):
                st.markdown(user_msg["content"])
            with st.chat_message("assistant"):
                st.markdown(asst_msg["content"])
                if asst_msg.get("used"):
                    st.markdown("**Sources**")
                    show_matches(asst_msg["used"],
                                 key_prefix=f"chat{user_i + 1}_")
                chat_cost_caption(asst_msg.get("tokens"), price,
                                  chat_in_price, chat_out_price)

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
            if cols[3].button("Preview",
                              key=f"preview_policy_{entry['slug']}"):
                _policy_dialog(entry, pdf_path.read_bytes())

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
    st.markdown("**The prompt in use** — the exact wording sent to "
                "the model on every answer, read live from the "
                "engine's configuration so this view cannot drift "
                "from what actually runs.")
    st.markdown("System prompt — the same on every request:")
    st.code(config.SYSTEM_PROMPT)
    st.markdown("User message — built fresh each time. The "
                "employee's words go in at the top; the passages the "
                "retrieval stage found are filled in underneath:")
    st.code(
        "Message from an employee:\n"
        "<the employee's exact words>\n\n"
        "Policy passages retrieved:\n"
        "[Source 1] <policy title> - <section> (page <n>)\n"
        "<passage text, filled in from the knowledge base>\n\n"
        "[Source 2] <policy title> - <section> (page <n>)\n"
        "<passage text, filled in from the knowledge base>")
    st.caption(f"Sent to {config.CHAT_MODEL} at temperature 0.2. "
               "When no passage passes the threshold and K filters, "
               "no prompt is built and the model is never called.")
    lab_text = st.text_area("Message to inspect",
                            value=SAMPLES[0]["complaint"], height=110,
                            key="lab_complaint")
    if st.button("Run the stages", key="lab_run"):
        if collection is None or not api_key:
            st.error("Needs the knowledge base and an OpenAI key.")
        else:
            try:
                client, result, _query_tokens = run_retrieval(lab_text)
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
