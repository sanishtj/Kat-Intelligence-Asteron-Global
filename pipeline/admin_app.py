"""Tenant admin console (LOCAL ONLY) - plays the Asteron tenant admin.

Upload policy PDFs, then run the pipeline steps in order. This app is
never deployed publicly: uploads and the OpenAI key stay on the tenant's
machine, exactly the split Sanish chose for the PoC. The public gets the
read-only playground app instead.

Run:  streamlit run admin_app.py
"""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
import config
from embedder import estimate, load_chunks

_GK_LOGO = PIPELINE_DIR.parent / "assets" / "guardkat-logo.png"
st.set_page_config(page_title="Kat Intelligence - Tenant Admin",
                   page_icon=str(_GK_LOGO) if _GK_LOGO.exists() else "🛡️")
_brand = st.columns([1, 6])
if _GK_LOGO.exists():
    _brand[0].image(str(_GK_LOGO))
with _brand[1]:
    st.title("Kat Intelligence · Tenant Admin")
    st.caption("A GuardKat concept · Asteron Global (demonstration "
               "tenant). Uploads stay on this machine; only the "
               "finished knowledge base ships with the public "
               "playground.")

st.header("1 · Upload policy PDFs")
uploads = st.file_uploader("Policy documents (PDF only)",
                           type=["pdf"], accept_multiple_files=True)
if uploads and st.button("Save uploads"):
    for upload in uploads:
        dest = config.POLICIES_DIR / upload.name
        dest.write_bytes(upload.getbuffer())
    st.success(f"Saved {len(uploads)} file(s) to the policies folder. "
               "Add them to manifest.json, then run the steps below.")

st.header("2 · Build the knowledge base")
st.write("Extract reads each PDF page by page, keeping the page number "
         "and section for every chunk. Embed needs the OpenAI key in "
         ".streamlit/secrets.toml and charges only for new text.")
st.markdown("**Settings in use** (from config.py — the same values "
            "the public playground displays)")
st.dataframe(
    [{"Setting": k, "Value": str(v)} for k, v in [
        ("Chunk size (max characters)", config.CHUNK_MAX_CHARS),
        ("Overlap between chunks (characters)",
         config.CHUNK_OVERLAP_CHARS),
        ("Smallest tail kept (characters)", config.CHUNK_MIN_TAIL_CHARS),
        ("Shortest text kept after cleaning (characters)",
         config.MIN_CLEANED_CHARS),
        ("Embedding model", config.EMBEDDING_MODEL),
        ("Numbers per chunk (vector size)", config.EMBEDDING_DIMENSIONS),
        ("Embedding price (USD per 1M tokens)",
         config.EMBEDDING_PRICE_USD_PER_1M_DEFAULT),
        ("Knowledge base collection", config.COLLECTION_NAME),
    ]],
    hide_index=True, use_container_width=True)
st.markdown("**Cost calculator** — what the Embed step will charge, "
            "shown before it charges anything")
price = st.number_input(
    "Embedding price (USD per 1M tokens)", min_value=0.0,
    value=float(config.EMBEDDING_PRICE_USD_PER_1M_DEFAULT),
    step=0.01, format="%.4f",
    help="The price is an input, never a hidden constant. The Embed "
         "step runs at exactly this price.")
try:
    _chunks = load_chunks()
except Exception:
    _chunks = []
if _chunks:
    _est = estimate(_chunks, price)
    _cols = st.columns(3)
    _cols[0].metric("Chunks to embed", _est["chunks_to_embed"])
    _cols[1].metric("Estimated tokens", _est["tokens"])
    _cols[2].metric("Estimated cost", f"${_est['est_cost_usd']:.4f}")
    st.caption(f"Model: {config.EMBEDDING_MODEL} · "
               f"{_est['chunks_skipped']} chunk(s) already embedded "
               "and skipped — re-runs never pay twice for the same "
               "text.")
else:
    st.info("Run the Chunk step to see the embedding estimate here.")

STEPS = [("Extract", "extract.py"), ("Clean", "cleaner.py"),
         ("Chunk", "chunker.py"), ("Embed", "embedder.py"),
         ("Store in ChromaDB", "kb_store.py")]
for label, script in STEPS:
    if st.button(f"Run: {label}"):
        env = None
        if script == "embedder.py":
            env = {**os.environ, "EMBEDDING_PRICE_USD": str(price)}
        with st.spinner(f"Running {label}..."):
            proc = subprocess.run(
                [sys.executable, script], cwd=PIPELINE_DIR,
                capture_output=True, text=True, env=env)
        st.code(proc.stdout or "(no output)")
        if proc.returncode != 0:
            st.error(proc.stderr[-2000:]
                     or "The step failed — see the message in the "
                        "output above.")

st.header("3 · Knowledge base status")
if config.CHROMA_DIR.exists():
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        coll = client.get_collection(config.COLLECTION_NAME)
        st.metric("Chunks in ChromaDB", coll.count())
    except Exception:
        st.info("ChromaDB folder exists but the collection is not built yet.")
else:
    st.info("Not built yet. Run the steps above in order.")
