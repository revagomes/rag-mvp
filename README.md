# rag-mvp

A minimal, **local-first** Retrieval-Augmented Generation (RAG) server for
testing a RAG pipeline against your own documents. It runs entirely on your
machine with **no API keys required** by default.

- **Embeddings:** `sentence-transformers` (local) or OpenAI
- **Vector store:** ChromaDB, persisted to disk
- **LLM (answers):** `none` (retrieval only), local **Ollama**, or OpenAI
- **Interfaces:** a FastAPI HTTP server and a CLI

The defaults (local embeddings + `none` LLM) let you test *retrieval quality*
immediately without downloading a language model. Point it at Ollama or OpenAI
later to get generated answers.

---

## Requirements

- Python 3.10–3.12 (a 3.12 virtualenv is recommended; some ML wheels lag on 3.14)
- [`uv`](https://docs.astral.sh/uv/) (or plain `pip`)

## Setup

```bash
cd rag-mvp
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

The first query/ingest downloads the local embedding model
(`all-MiniLM-L6-v2`, ~90 MB) once and caches it.

Optional configuration lives in `.env` (copy from `.env.example`). Everything
has defaults, so `.env` is not required.

---

## Quick start (CLI)

```bash
# 1. Ingest the bundled sample documents
python -m rag_mvp.cli ingest sample_docs/

# 2. Ask a question (retrieval-only by default)
python -m rag_mvp.cli query "What is RAG and why is it useful?"

# 3. Inspect configuration and store size
python -m rag_mvp.cli info

# 4. Clear the store
python -m rag_mvp.cli reset
```

With the default `none` LLM backend, `query` prints the retrieved chunks and
their similarity scores — ideal for checking whether retrieval is finding the
right passages.

---

## Quick start (HTTP server)

```bash
uvicorn rag_mvp.server:app --reload
```

Then, from another terminal:

```bash
# Health / config
curl localhost:8000/health

# Ingest a file or directory that exists on the server's disk
curl -X POST localhost:8000/ingest/path \
  -H 'Content-Type: application/json' \
  -d '{"path": "sample_docs"}'

# Ingest raw text
curl -X POST localhost:8000/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is Paris.", "source": "geo"}'

# Query
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which planet is the largest?", "top_k": 3}'
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## Getting generated answers

### Option A — Ollama (local, free)

1. Install Ollama and pull a model:
   ```bash
   ollama pull llama3.2
   ```
2. In `.env`:
   ```
   RAG_LLM_BACKEND=ollama
   RAG_OLLAMA_MODEL=llama3.2
   ```

### Option B — OpenAI

In `.env`:
```
RAG_LLM_BACKEND=openai
RAG_OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

You can also switch embeddings to OpenAI with `RAG_EMBEDDING_BACKEND=openai`.
If you change the embedding backend or model, run `reset` and re-ingest, since
vectors from different models are not comparable.

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `RAG_EMBEDDING_BACKEND` | `local` | `local` or `openai` |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | local sentence-transformers model |
| `RAG_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RAG_LLM_BACKEND` | `none` | `none`, `ollama`, or `openai` |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `RAG_OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `RAG_OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_API_KEY` | – | required for any OpenAI backend |
| `RAG_CHUNK_SIZE` | `800` | characters per chunk |
| `RAG_CHUNK_OVERLAP` | `120` | overlap between chunks |
| `RAG_TOP_K` | `4` | chunks retrieved per query |
| `RAG_STORAGE_DIR` | `./storage` | where Chroma persists data |
| `RAG_COLLECTION_NAME` | `documents` | Chroma collection name |

---

## Project layout

```
rag-mvp/
├── rag_mvp/
│   ├── config.py       # settings (env / .env)
│   ├── ingest.py       # load + chunk documents
│   ├── embeddings.py   # local / OpenAI embedders
│   ├── vectorstore.py  # ChromaDB persistence + search
│   ├── llm.py          # none / Ollama / OpenAI answer generation
│   ├── pipeline.py     # orchestrates the full RAG flow
│   ├── schemas.py      # API request/response models
│   ├── server.py       # FastAPI app
│   └── cli.py          # command-line interface
├── sample_docs/        # example documents to ingest
├── pyproject.toml
└── .env.example
```

---

## Testing

```bash
uv pip install -e ".[dev]"
pytest                       # run the suite
pytest --cov --cov-report=term-missing   # with coverage
```

The suite is deliberately focused on the **deterministic core** and runs fully
offline — no model downloads, no network. Fakes for the embedder, vector store,
and LLM are injected via `RagPipeline(..., embedder=, store=, llm=)`.

Coverage is ~94%. The uncovered lines are, by design, the thin wrappers over
external systems:

- `embeddings.py` — the OpenAI embedder and the real `sentence-transformers`
  model load. Verified by running against the actual model/API.
- `llm.py` — the OpenAI chat backend (the Ollama backend *is* tested with a
  faked HTTP layer). Verified against a live Ollama server.
- a PDF text-extraction line and one branch in the chunker's inner loop.

These are I/O-bound integration points where a unit test would only assert that
mocks return what they were told to. They are exercised by the manual
CLI/Ollama runs documented above instead. See `[tool.coverage.run].omit` in
`pyproject.toml` (the CLI is excluded for the same reason).

## How it works

1. **Ingest** — documents are loaded (`.txt`, `.md`, `.pdf`), normalized, and
   split into overlapping character windows that prefer word boundaries.
2. **Embed** — each chunk is turned into a normalized vector.
3. **Store** — vectors + text + metadata go into a persistent Chroma collection
   using cosine distance.
4. **Retrieve** — a question is embedded and the top-k most similar chunks are
   fetched, each with a similarity score (1.0 = identical).
5. **Generate** — retrieved chunks are formatted into a context block and passed
   to the configured LLM (or returned as-is when the backend is `none`).


---

## License

This project is licensed under the **GNU General Public License v2.0 or later**
(`GPL-2.0-or-later`). See the [LICENSE](LICENSE) file for the full text.
