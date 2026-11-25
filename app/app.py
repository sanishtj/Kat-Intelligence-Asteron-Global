"""Kat Intelligence - Asteron Global playground (the public app).

Home is the playground: describe a workplace complaint as if you were
an Asteron employee, and Kat Intelligence finds the company policies
that match it, explains why, and opens the policy PDF at the exact
page, with the passage highlighted.

Tabs: Chat - Playground - Policies - Magic - Prompt.
"""

import base64
import html
import io
import json
import os
import sys
import tomllib
from datetime import timedelta
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
                     ("chat_messages", []), ("chat_pending", None),
                     ("lab_result", None), ("lab_error", None)):
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
  max-width: 880px; text-align: center;}
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
/* A prompt piece on the Prompt tab: the exact text the model
   receives, line breaks kept, long lines wrapped inside the
   box instead of scrolling sideways for ever. */
.gk-prompt {background: #f7f7f5; border: 1px solid rgba(38, 50, 56, 0.14);
  border-radius: 8px; padding: 12px 14px; margin-bottom: 12px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.85rem; line-height: 1.55; color: #263238;
  white-space: pre-wrap; overflow-wrap: break-word;}
/* Top spacing: Streamlit reserves a tall band at the top of the
   main area and above the sidebar header. Pull the content up,
   but the main padding must still clear Streamlit's own fixed
   top header (about 3.5rem, the bar holding Deploy): at 1.6rem
   the brand logos slid underneath it and looked cut off. 4.25rem
   clears it with a small gap and is still well under the 6rem
   default. */
[data-testid="stMainBlockContainer"] {padding-top: 4.25rem;}
[data-testid="stAppViewBlockContainer"] {padding-top: 4.25rem;}
[data-testid="stSidebarHeader"] {padding-top: 0.4rem;
  padding-bottom: 0;}
/* Tabs that read as tabs: larger labels, boxed tops, and the
   selected tab white with the brand amber line under it.
   Streamlit 1.67 tabs are react-aria, not BaseWeb: each tab is
   [data-testid="stTab"] and the selected one carries
   data-selected / aria-selected. The old [data-baseweb="tab"]
   selectors matched nothing, which is why the previous styling
   round produced no visible change. */
[data-testid="stTabs"] [data-testid="stTab"]
  {font-size: 1.08rem !important; font-weight: 600;
  color: #546e7a !important; padding: 10px 20px;
  background: rgba(38, 50, 56, 0.06) !important;
  border: 1px solid rgba(38, 50, 56, 0.18); border-bottom: none;
  border-radius: 10px 10px 0 0; margin-right: 6px;}
[data-testid="stTabs"] [data-testid="stTab"][data-selected="true"],
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"]
  {background: #ffffff !important; color: #263238 !important;
  box-shadow: inset 0 -3px 0 #d98e2b;}
/* The chat box is the centrepiece of the Chat tab: a framed,
   tinted panel with a large input, not a bare text box. */
[data-testid="stForm"] {border: 2px solid #d98e2b;
  border-radius: 14px; background: #fffaf1;
  padding: 14px 16px 6px;
  box-shadow: 0 2px 12px rgba(38, 50, 56, 0.10);}
[data-testid="stForm"] [data-testid="stTextInput"] input
  {font-size: 1.1rem; padding: 0.65rem 0.8rem;}
/* How-it-was-built stepper: a row of icon circles joined by a
   line, the current step lit in brand amber, and a card under it
   with a fifteen-second progress bar while the walk auto-plays. */
.gk-steps {display: flex; margin: 0.8rem 0 0.4rem;}
.gk-step {flex: 1; display: flex; flex-direction: column;
  align-items: center; position: relative;}
.gk-step:before {content: ""; position: absolute; top: 26px;
  left: -50%; width: 100%; height: 3px; background: #cfd8dc;}
.gk-step:first-child:before {display: none;}
.gk-step.done:before, .gk-step.now:before {background: #d98e2b;}
.gk-ico {width: 52px; height: 52px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; background: #eceff1; border: 2px solid #cfd8dc;
  position: relative; z-index: 1;}
.gk-step.done .gk-ico {background: #d98e2b; border-color: #d98e2b;}
.gk-step.now .gk-ico {background: #ffffff; border-color: #d98e2b;
  box-shadow: 0 0 0 5px rgba(217, 142, 43, 0.25);
  transform: scale(1.12);}
.gk-lab {margin-top: 6px; font-size: 0.8rem; font-weight: 600;
  color: #78909c;}
.gk-step.now .gk-lab {color: #263238;}
.gk-card {border: 2px solid rgba(217, 142, 43, 0.55);
  border-radius: 14px; background: #fffaf1; padding: 20px 22px;
  margin-top: 10px; margin-bottom: 16px; text-align: center;}
.gk-card-ico {font-size: 2.4rem; line-height: 1;}
.gk-card-step {margin-top: 8px; font-size: 0.78rem;
  letter-spacing: 1.5px; color: #b87718; font-weight: 700;
  text-align: center;}
.gk-card-title {font-size: 1.45rem; font-weight: 800;
  color: #263238; margin-top: 2px; text-align: center;}
.gk-card p.gk-card-text {color: #455a64; max-width: 640px;
  margin: 8px auto 0; text-align: center !important;}
.gk-card-cta {margin-top: 16px; font-size: 1.05rem;
  font-weight: 700; color: #b87718; text-align: center;}
.gk-tiles {display: flex; gap: 12px; justify-content: center;
  margin-top: 16px; flex-wrap: wrap;}
.gk-tile {background: #ffffff;
  border: 1px solid rgba(217, 142, 43, 0.35); border-radius: 12px;
  padding: 10px 18px; min-width: 130px;}
.gk-num {font-size: 1.6rem; font-weight: 800; color: #263238;}
.gk-cap {font-size: 0.78rem; color: #607d8b;}
.gk-bar {height: 5px; background: rgba(38, 50, 56, 0.08);
  border-radius: 3px; overflow: hidden; margin-top: 16px;}
.gk-bar > div {height: 100%; background: #d98e2b;
  animation-duration: 15s; animation-timing-function: linear;
  animation-fill-mode: forwards;}
/* Three identical keyframe sets on purpose: the stepper names
   one per run, alternating, because a browser will not restart
   an animation on an element it kept from the previous run -
   with a single name the bar filled once and then sat frozen
   at the end state through every autoplay step. */
@keyframes gkfillA {from {width: 0;} to {width: 100%;}}
@keyframes gkfillB {from {width: 0;} to {width: 100%;}}
@keyframes gkfillR {from {width: 0;} to {width: 100%;}}
/* Stepper controls, deliberately prominent: big bold labels,
   amber outlines, a warm fill on Previous and Next, and Play or
   Pause as a filled primary button, so nobody misses them. */
.st-key-built_prev button, .st-key-built_next button {
  font-size: 1.05rem; font-weight: 700; color: #263238;
  background: #fff3dd; border: 2px solid #d98e2b;
  border-radius: 10px; padding: 0.55rem 0.5rem;}
.st-key-built_prev button:hover, .st-key-built_next button:hover {
  background: #ffe7bd; color: #263238; border-color: #b87718;}
.st-key-built_pause button {font-size: 1.05rem; font-weight: 700;
  border: 2px solid #b87718; border-radius: 10px;
  padding: 0.55rem 0.5rem;}
/* The seven steps are full-width tile buttons, one per column.
   Streamlit's own layout stretches each tile across its column
   and centres its label, so the row reads as one connected bar
   and nothing depends on wrapper widths or absolute positioning.
   The state colours (done in amber, the current one ringed, the
   rest grey) are emitted per run by the stepper itself. */
[class*="st-key-built_step_"] button {width: 100%;
  font-size: 0.98rem; font-weight: 700; border-radius: 12px;
  padding: 0.65rem 0.2rem; background: #eceff1;
  border: 2px solid #cfd8dc; color: #455a64;
  position: relative; z-index: 1;}
[class*="st-key-built_step_"] button:hover {
  border-color: #d98e2b; color: #263238;}
/* Each step's icon sits in a small white circle, so the icon
   stays readable when its tile fills amber behind it. */
[class*="st-key-built_step_"] button [data-testid="stIconEmoji"]
  {display: inline-flex; align-items: center;
   justify-content: center; width: 30px; height: 30px;
   border-radius: 50%; background: #ffffff; flex-shrink: 0;}
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
        '<p class="gk-sub" style="text-align:center; '
        'margin-left:auto; margin-right:auto">'
        'GuardKat\u2019s document intelligence, '
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
    ["💬 Chat", "📋 Playground", "📄 Policies", "✨ Magic",
     "📝 Prompt"])

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
def _built_steps(stats: dict) -> list[dict]:
    """The seven steps of the whole system, with the real numbers
    from the knowledge base on disk. Icons included: this walk is
    meant to be watched, not read."""
    _p = pipeline_settings()
    docs = stats.get("documents", "—")
    pages = stats.get("pages", "—")
    segments = stats.get("segments", "—")
    chunks = stats.get("chunks", "—")
    tokens = stats.get("tokens")
    token_txt = f"{tokens:,}" if isinstance(tokens, int) else "—"
    cost_txt = (f"${tokens / 1_000_000 * price:.4f}"
                if isinstance(tokens, int) else "—")
    return [
        {"icon": "📥", "label": "Collect",
         "title": "Collect the documents",
         "text": "Asteron Global's policy library, gathered in one "
                 "place: ethics, bribery, harassment, data protection "
                 "and the rest of the set.",
         "tiles": [(str(docs), "policy documents"),
                   (str(pages), "pages in total")]},
        {"icon": "📖", "label": "Read",
         "title": "Read every page",
         "text": "Each PDF is read page by page, in order. No AI is "
                 "used in this step, and every passage keeps the page "
                 "number it came from, which is what makes the "
                 "exact-page preview possible later.",
         "tiles": [(str(pages), "pages read"),
                   (str(segments), "text segments found"),
                   ("0", "AI calls in this step")]},
        {"icon": "🧹", "label": "Clean",
         "title": "Clean the text",
         "text": "Whitespace is tidied and the letterhead and footer "
                 "bands are dropped. The policy's own words are the "
                 "source of truth: nothing is added, removed or "
                 "rewritten.",
         "tiles": [("0", "words changed"),
                   (str(segments), "segments kept")]},
        {"icon": "✂️", "label": "Chunk",
         "title": "Cut the text into chunks",
         "text": "Segments are packed in order into small passages "
                 "that never cross a section boundary, so every chunk "
                 "is one complete idea, carrying its document, "
                 "section and page range.",
         "tiles": [(str(chunks), "chunks created"),
                   (f"{_p.get('chunk_max', '—'):,}"
                    if isinstance(_p.get("chunk_max"), int)
                    else str(_p.get("chunk_max", "—")),
                    "max characters each"),
                   (str(_p.get("chunk_overlap", "—")),
                    "characters of overlap")]},
        {"icon": "🔢", "label": "Embed",
         "title": "Turn chunks into numbers",
         "text": "text-embedding-3-small turns each chunk into a "
                 "list of numbers that captures its meaning, so "
                 "chunks about similar things sit close together. "
                 "Unchanged chunks are skipped on re-runs, so "
                 "updates only pay for what changed.",
         "tiles": [("1,536", "numbers per chunk"),
                   (token_txt, "tokens in total"),
                   (cost_txt, "one-time cost")]},
        {"icon": "🗄️", "label": "Store",
         "title": "Store them in ChromaDB",
         "text": "Chunks and their number lists are saved in a "
                 "persistent vector store, one collection for the "
                 "company, ready to be searched by meaning rather "
                 "than by keywords.",
         "tiles": [(str(chunks), "vectors stored"),
                   ("1", "collection"),
                   ("~2 MB", "whole knowledge base")]},
        {"icon": "💬", "label": "RAG",
         "title": "RAG — the answer, with the exact page",
         "text": "This last step has a name: RAG, retrieval-augmented "
                 "generation. A message is turned into numbers the "
                 "same way and matched against the store. The best "
                 "passages go to the chat model, which may only use "
                 "them, and the policy opens on the exact page with "
                 "the passage highlighted. If nothing matches, the "
                 "answer honestly says so.",
         "tiles": [(f"{int(config.THRESHOLD_DEFAULT * 100)}%",
                    "minimum match to count"),
                   (str(config.MAX_K_DEFAULT), "passages at most"),
                   ("0", "invented answers")]},
    ]


def _built_nav(target: int) -> None:
    st.session_state["built_step"] = target
    st.session_state["built_nav"] = True
    st.session_state["built_rag_held"] = False


def _built_jump(target: int) -> None:
    """An icon click: go straight to that step and pause the walk
    there, so the reader is not carried away mid-read."""
    _built_nav(target)
    st.session_state["built_paused"] = True


def _built_toggle_pause() -> None:
    st.session_state["built_paused"] = (
        not st.session_state.get("built_paused", False))
    st.session_state["built_nav"] = True


@st.fragment(run_every=timedelta(seconds=15))
def built_stepper() -> None:
    """The auto-playing walk through the seven steps. The fragment
    re-runs every fifteen seconds and advances one step, looping for
    ever; the buttons set built_nav so a click never also triggers
    an advance on the same run."""
    steps = _built_steps(kb_stats())
    total = len(steps)
    if "built_step" not in st.session_state:
        st.session_state["built_step"] = 0
        st.session_state.pop("built_nav", None)
    elif st.session_state.pop("built_nav", False):
        pass
    elif not st.session_state.get("built_paused", False):
        _cur = st.session_state["built_step"] % total
        if (_cur == total - 1
                and not st.session_state.get("built_rag_held")):
            # The RAG step stays twice as long as the others: hold
            # it for one extra interval before the loop restarts.
            st.session_state["built_rag_held"] = True
        else:
            st.session_state["built_rag_held"] = False
            st.session_state["built_step"] = (_cur + 1) % total
    idx = st.session_state["built_step"] % total
    step = steps[idx]
    paused = st.session_state.get("built_paused", False)
    # The seven steps are tile buttons, one per column, each
    # carrying its icon and its name: one click jumps straight to
    # that step and pauses the walk there, so a reader can sit on
    # the step they care about. Their state colours (done in
    # amber, the current one ringed, the rest grey) are emitted
    # per run because they follow the current step.
    state_css = []
    for i in range(total):
        if i < idx:
            look = ("background:#d98e2b;border-color:#d98e2b;"
                    "color:#ffffff;")
        elif i == idx:
            look = ("background:#ffffff;border-color:#d98e2b;"
                    "color:#263238;"
                    "box-shadow:0 0 0 5px rgba(217,142,43,0.25);")
        else:
            look = ("background:#eceff1;border-color:#cfd8dc;"
                    "color:#455a64;")
        state_css.append(
            f'[class*="st-key-built_step_{i}"] button {{{look}}}')
        # The same look on hover: Streamlit's own hover style
        # would otherwise wash the tile translucent and the
        # joining line behind it shows through, cutting across
        # the tile's text. Hover changes nothing but the cursor.
        state_css.append(
            f'[class*="st-key-built_step_{i}"] button:hover '
            f'{{{look}}}')
    # The joining line lives on the columns row itself: a grey
    # track running from the first tile's centre to the last
    # tile's, with an amber fill over it from the start to the
    # current tile's centre. The tiles paint above the line, so
    # it shows in the gaps between them and reads as one line
    # joining all seven steps. (An earlier version painted the
    # line as segments behind each icon; when the icons became
    # real buttons the segments anchored to the wrong elements
    # and the line vanished. The row is found by what it holds,
    # which no other row on the page does.)
    _row = ('[data-testid="stHorizontalBlock"]'
            ':has([class*="st-key-built_step_0"])')
    _c = 50.0 / total
    state_css.append(f"{_row} {{position: relative;}}")
    state_css.append(
        f'{_row}::before {{content: ""; position: absolute; '
        f"top: 50%; transform: translateY(-50%); "
        f"left: {_c:.3f}%; right: {_c:.3f}%; height: 3px; "
        f"background: #cfd8dc; z-index: 0;}}")
    state_css.append(
        f'{_row}::after {{content: ""; position: absolute; '
        f"top: 50%; transform: translateY(-50%); "
        f"left: {_c:.3f}%; width: {idx * 100.0 / total:.3f}%; "
        f"height: 3px; background: #d98e2b; z-index: 0;}}")
    st.markdown(f"<style>{''.join(state_css)}</style>",
                unsafe_allow_html=True)
    icon_cols = st.columns(total)
    for i, (col, s) in enumerate(zip(icon_cols, steps)):
        with col:
            st.button(s["label"], icon=s["icon"],
                      key=f"built_step_{i}", width="stretch",
                      on_click=_built_jump, args=(i,),
                      help=f"Go to step {i + 1}: {s['label']}")
    tiles = "".join(
        f'<div class="gk-tile"><div class="gk-num">{num}</div>'
        f'<div class="gk-cap">{cap}</div></div>'
        for num, cap in step["tiles"])
    cta = (f'<div class="gk-card-cta">{step["cta"]}</div>'
           if step.get("cta") else "")
    # The bar's fill is a CSS animation, and on the fragment's
    # timed re-runs the browser keeps the bar's element whenever
    # its HTML is unchanged - a kept element never restarts its
    # animation, so the bar filled once and sat at the end state
    # through the whole autoplay. Alternating between two
    # identical keyframe names on every run forces the restart.
    # The RAG step keeps one fixed name and a 30 second duration,
    # so its bar runs unbroken through the extra hold interval.
    _tick = st.session_state.get("built_tick", 0) + 1
    st.session_state["built_tick"] = _tick
    if paused:
        bar = ""
    elif idx == total - 1:
        bar = ('<div class="gk-bar"><div style="animation-name: '
               'gkfillR; animation-duration: 30s"></div></div>')
    else:
        _fill = "gkfillA" if _tick % 2 == 0 else "gkfillB"
        bar = (f'<div class="gk-bar"><div style="animation-name: '
               f'{_fill}"></div></div>')
    st.markdown(
        f'<div class="gk-card"><div class="gk-card-ico">'
        f'{step["icon"]}</div>'
        f'<div class="gk-card-step">Step {idx + 1} of {total} · '
        f'{step["label"]}</div>'
        f'<div class="gk-card-title">{step["title"]}</div>'
        f'<p class="gk-card-text">{step["text"]}</p>'
        f'{cta}<div class="gk-tiles">{tiles}</div>{bar}</div>',
        unsafe_allow_html=True)
    if idx == total - 1:
        # On the RAG step there is no call-to-action text. Instead a
        # large pointing hand bobs up and down directly underneath
        # the Prompt tab itself, so the next move is obvious without
        # a word of instruction. The hand belongs to the Magic tab:
        # the style stays in the page after a tab switch (the tab
        # panel only hides), so the rule is gated on the Magic tab
        # being the selected one, and the hand vanishes the moment
        # any other tab is picked.
        # The tabs in this Streamlit version are not buttons:
        # they are divs carrying data-testid "stTab", five of
        # them in a row, so the fifth one is the Prompt tab.
        # (Verified in a real browser; an earlier version of
        # this rule aimed at tab buttons, which do not exist
        # here, and the hand never rendered.)
        _lists = ('[data-testid="stTabList"], '
                  '[data-baseweb="tab-list"], [role="tablist"]')
        _tabs = '[data-testid="stTab"]:nth-of-type(5)'
        _magic = '[data-testid="stTab"]:nth-of-type(4)'
        _hands = ", ".join(
            f'{_magic}[{attr}="true"] + {_tabs}::after'
            for attr in ("aria-selected", "data-selected"))
        st.markdown(
            "<style>"
            f"{_lists} {{overflow: visible !important;}}"
            f"{_tabs} {{position: relative;}}"
            f"{_hands} {{content: \"👆\"; position: absolute; "
            "top: 100%; left: 50%; font-size: 42px; line-height: 1; "
            "z-index: 50; pointer-events: none; "
            "animation: gkbob 0.55s ease-in-out infinite alternate;}"
            "@keyframes gkbob {"
            "from {transform: translate(-50%, 0);}"
            "to {transform: translate(-50%, 10px);}}"
            "</style>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.button("← Previous", key="built_prev", width="stretch",
              on_click=_built_nav, args=((idx - 1) % total,))
    c2.button("▶ Play" if paused else "⏸ Pause", key="built_pause",
              width="stretch", type="primary",
              on_click=_built_toggle_pause)
    c3.button("Next →", key="built_next", width="stretch",
              on_click=_built_nav, args=((idx + 1) % total,))


with tab_pipeline:
    st.subheader("How the knowledge base was built")
    st.write("The whole system, in seven steps, from the documents "
             "arriving to an answer opening the exact policy page. "
             "It plays on its own, one step every fifteen seconds, or "
             "step through it yourself with the buttons. Clicking any "
             "step icon jumps straight to it and pauses the walk.")
    built_stepper()
    st.caption("In production this same design swaps the local "
               "ChromaDB folder for Azure AI Search; the chunks and "
               "the embeddings move across unchanged.")
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
    with st.expander("All the settings, for the curious"):
        st.markdown("Read live from the configuration files, so "
                    "what you see here is exactly what runs.")
        _set_cols = st.columns(2)
        with _set_cols[0]:
            st.markdown("*Preparing the documents*")
            st.dataframe(
                [{"Setting": k, "Value": str(v)} for k, v in _prep],
                hide_index=True, use_container_width=True)
        with _set_cols[1]:
            st.markdown("*Finding the answers*")
            st.dataframe(
                [{"Setting": k, "Value": str(v)} for k, v in _find],
                hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- lab
def _run_lab() -> None:
    """Prompt tab run: retrieve, fill the prompt's placeholders,
    generate the answer, and keep everything in session state so
    the filled prompt and the results stay on the page."""
    text = st.session_state.get("lab_complaint", "")
    st.session_state["lab_error"] = None
    if collection is None or not api_key:
        st.session_state["lab_result"] = None
        st.session_state["lab_error"] = (
            "Needs the knowledge base and an OpenAI key.")
        return
    try:
        client, result, _query_tokens = run_retrieval(text)
        answer = (explain.explain(client, text, result["used"])
                  if result["used"] else None)
        st.session_state["lab_result"] = {
            "text": text, "candidates": result["candidates"],
            "used": result["used"], "answer": answer}
    except Exception as exc:
        st.session_state["lab_result"] = None
        st.session_state["lab_error"] = (
            f"Something went wrong: {exc}")


def prompt_block(text: str) -> None:
    """Show one piece of the prompt exactly as the model
    receives it, wrapped inside its box (st.code would give
    it a sideways scroller that runs on for ever)."""
    st.markdown(f'<div class="gk-prompt">{html.escape(text)}</div>',
                unsafe_allow_html=True)


with tab_lab:
    st.subheader("The prompt — how every answer is built")
    st.write("Every answer is a single prompt: the standing "
             "instructions, the grounding rules inside them, the "
             "employee's message, and the policy passages retrieved "
             "for that message. The message box below sits inside "
             "the prompt itself. Type a message, run it, and the "
             "retrieved passages fill into their placeholders. The "
             "results appear underneath.")
    _run = st.session_state.get("lab_result")
    with st.container(border=True):
        st.markdown("**🧭 System prompt** — the standing "
                    "instructions. This is the model's role, the same "
                    "on every request, read live from the engine's "
                    "configuration so what you see is exactly what "
                    "runs:")
        _sys_head, _sys_sep, _sys_tail = (
            config.SYSTEM_PROMPT.partition("Using ONLY"))
        if _sys_sep:
            prompt_block(_sys_head)
            st.markdown("**🛡️ Grounding rules** — the part of the "
                        "system prompt that keeps the answer honest: "
                        "only the passages may be used, cite each "
                        "one, invent nothing, and say so when the "
                        "passages do not answer. Shown separately so "
                        "they are easy to see; the model receives "
                        "them inside the system prompt, word for "
                        "word:")
            prompt_block("Using ONLY" + _sys_tail)
        else:
            prompt_block(config.SYSTEM_PROMPT)
        st.markdown("**💬 Message from an employee (user "
                    "prompt)** — this box is part of the prompt. "
                    "Whatever is typed here goes to the model word "
                    "for word, under that heading:")
        lab_text = st.text_area(
            "Message from an employee",
            value=SAMPLES[0]["complaint"], height=110,
            key="lab_complaint", label_visibility="collapsed")
        st.markdown("**📚 Policy passages retrieved** — the "
                    "placeholders underneath the message. Retrieval "
                    "searches the ChromaDB knowledge base and fills "
                    "them in, fresh every time the prompt is built:")
        if _run:
            _n = len(_run["used"])
            st.markdown(
                f"**{_n} policy passage{'s' if _n != 1 else ''} "
                "used for this answer** — however many passed "
                "the threshold and K filters on this run, counted "
                "from the retrieval itself, never a fixed number:")
        if _run and _run["used"]:
            prompt_block("Policy passages retrieved:\n"
                         + explain.build_context(_run["used"]))
        elif _run:
            prompt_block("Policy passages retrieved:\n"
                         "<nothing passed the threshold and K "
                         "filters, so the prompt was never built "
                         "and the model was never called>")
        else:
            prompt_block("Policy passages retrieved:\n"
                         "<the passages retrieval finds will fill "
                         "in here, one after another, each with "
                         "its policy, section and page>")
        if _run and _run["text"] != lab_text:
            st.caption("The passages above belong to the message "
                       "you last ran. You have changed the message "
                       "since; run it again to refill the prompt.")
        st.button("Run it — fill the prompt and get the answer",
                  key="lab_run", on_click=_run_lab, type="primary",
                  width="stretch")
    st.caption(f"Sent to {config.CHAT_MODEL} at temperature 0.2, "
               "using the sidebar controls. When no passage passes "
               "the threshold and K filters, no prompt is built and "
               "the model is never called.")
    if st.session_state.get("lab_error"):
        st.error(st.session_state["lab_error"])
    if _run:
        st.markdown("**What retrieval found** — every candidate, "
                    "best first. ✓ = the passages that filled the "
                    "prompt above, after the threshold and K "
                    "filters.")
        _used_ids = {h["chunk_id"] for h in _run["used"]}
        _rows = [{
            "Used": "✓" if h["chunk_id"] in _used_ids else "—",
            "Match %": round(h["similarity"] * 100, 1),
            "Document": h["title"], "Section": h["section"],
            "Page": h["page_start"],
        } for h in _run["candidates"]]
        st.dataframe(_rows, use_container_width=True)
        if _run["used"]:
            st.markdown("**The answer** — written by the model "
                        "from the filled prompt above, and from "
                        "nothing else:")
            st.write(_run["answer"])
        else:
            st.info("No passages passed the filter, so no prompt "
                    "was built and the model was never called. "
                    "That restraint is deliberate.")

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
