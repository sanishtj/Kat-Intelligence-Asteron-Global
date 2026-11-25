# Kat Intelligence — Asteron Global (Proof of Concept)

**Kat Intelligence** is a GuardKat concept: a tenant's own policy
documents, made searchable at the moment a case is being handled.

A tenant admin uploads the company's policies (PDF). Each document is
read page by page, cleaned, chunked and embedded. When a case is
reported, the case handler is shown the policies that match the case,
with a preview that opens the PDF **at the exact page** the passage
came from, the passage highlighted. No more hunting through shared
drives while a case waits.

This repository is the public proof of concept. The tenant, **Asteron
Global**, is fictional, and all ten policies were written for this
demo, so everything here can be shown openly.

## The demo corpus

Ten Asteron Global policies in `policies-pdf/` (sources in
`policies-src/`, generator in `tools/`): Code of Conduct and Ethics,
Anti-Bribery and Corruption, Workplace Harassment Prevention, Data
Protection and Privacy, Information Security and Acceptable Use,
Conflicts of Interest, Health and Safety, Expense and Travel, Social
Media and External Communications, and Speak-Up and Whistleblowing.

## How it works

```
PDF upload (tenant admin, local)
  -> extraction, page by page, keeping page number + section
  -> cleaning
  -> chunking (structure-first: a chunk never crosses a section)
  -> embeddings (text-embedding-3-small, hash-checked so re-runs are free)
  -> ChromaDB folder (each chunk carries document, section, page range)
  -> a complaint is embedded and matched against the chunks
  -> an LLM explains the matches, grounded only in those chunks
  -> the matched PDF page is rendered with the passage highlighted
```

Design rules carried over from production thinking:

- **Citations are exact.** Every chunk knows its document, section and
  page range from the moment of extraction.
- **Weak matches show nothing.** A relevance threshold decides whether
  any policy is suggested at all; the honest "no policy matched" state
  is a feature, not a failure.
- **Tenants are isolated.** In production, each tenant's documents live
  in their own store and retrieval respects who may see the case. In
  this PoC there is one fictional tenant.
- **The store is a folder on purpose.** ChromaDB as a folder of files
  is enough for a proof of concept; the production design swaps it for
  Azure AI Search without changing the pipeline, because chunks and
  embeddings are portable.

## Layout

- `policies-src/` — the ten policy sources (plain text)
- `policies-pdf/` — the generated PDFs and `manifest.json`
- `tools/generate_policies.py` — builds the PDFs (reportlab)
- `pipeline/` — extraction, cleaning, chunking, embedding, Chroma store,
  plus `admin_app.py`, the local-only tenant upload console
- `assets/` — the Asteron Global logo

## Running the pipeline

```bash
cd pipeline
pip install -r requirements.txt
# put your OpenAI key in pipeline/.streamlit/secrets.toml
# (copy .streamlit/secrets.toml.example); the key is never committed
python extract.py     # PDFs -> per-page segments with section + page
python cleaner.py     # conservative whitespace cleanup
python chunker.py     # section-aware chunks with page metadata
python embedder.py    # embeddings, with a cost estimate before any call
python kb_store.py    # upsert into the ChromaDB folder
```

## Status

Proof of concept, built in phases: corpus and pipeline complete;
independent threshold and K filters (fixed K or dynamic K), the grounded
explanation and the highlighted page preview next; then the public
Streamlit playground and a test pass that sets the default threshold
and K from measured scores.
