"""Tenant admin console (LOCAL ONLY) - plays the Asteron tenant admin.

Upload policy PDFs, then run the pipeline steps in order. This app is
never deployed publicly: uploads and the OpenAI key stay on the tenant's
machine, exactly the split Sanish chose for the PoC. The public gets the
read-only playground app instead.

Run:  streamlit run admin_app.py
"""

import subprocess
import sys
from pathlib import Path

import streamlit as st

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
import config

st.set_page_config(page_title="Kat Intelligence - Tenant Admin",
                   page_icon="🛡️")
st.title("Kat Intelligence · Tenant Admin")
st.caption("Asteron Global (demonstration tenant). Uploads stay on this "
           "machine; only the finished knowledge base ships with the "
           "public playground.")

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

STEPS = [("Extract", "extract.py"), ("Clean", "cleaner.py"),
         ("Chunk", "chunker.py"), ("Embed", "embedder.py"),
         ("Store in ChromaDB", "kb_store.py")]
for label, script in STEPS:
    if st.button(f"Run: {label}"):
        with st.spinner(f"Running {label}..."):
            proc = subprocess.run(
                [sys.executable, script], cwd=PIPELINE_DIR,
                capture_output=True, text=True)
        st.code(proc.stdout or "(no output)")
        if proc.returncode != 0:
            st.error(proc.stderr[-2000:])

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
