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

## Preparing documents

The ingester accepts `.txt`, `.md`, and `.pdf`. If your source material is
saved web pages (`.mhtml` — the "Save as / Webpage, Single File" format from
Chrome/Edge, common for intranet, wiki, or SharePoint exports), convert them to
clean Markdown first. Raw MHTML is a MIME container full of quoted-printable
markup, inline CSS, and base64 images that would pollute embeddings.

The bundled converter extracts the main document, strips scripts/styles/nav
chrome, and produces Markdown that **preserves headings, lists, and
hyperlinks** (so retrieved chunks keep their `[text](url)` references):

```bash
# One-time: install the optional conversion dependencies
uv pip install -e ".[convert]"

# Convert a directory of .mhtml files (or a single file) into Markdown
python scripts/mhtml_to_markdown.py path/to/mhtml_dir path/to/output_dir

# Then ingest the generated Markdown as usual
python -m rag_mvp.cli ingest path/to/output_dir
```

The converter lives in `scripts/` as a standalone utility — it is not part of
the server runtime and has no effect unless you run it.

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
| `RAG_INGEST_ROOT` | `.` | filesystem root `/ingest/path` is confined to |
| `RAG_API_KEYS` | (empty) | comma-separated Bearer keys; empty disables auth |

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

## Security

This is an MVP intended for **local, single-user** use. It has been through a
lightweight hardening pass, but review the following before exposing it beyond
localhost.

### What is hardened
- **Path containment** — `/ingest/path` only accepts paths inside
  `RAG_INGEST_ROOT` (default: the current working directory). Traversal
  (`../`), absolute paths outside the root, and symlinks escaping the root are
  rejected with `403`. This prevents reading arbitrary files (e.g. `/etc/passwd`,
  `~/.ssh/id_rsa`) and exfiltrating them via `/query`.
- **Bounded inputs** — request fields have maximum lengths (text 1 MB, question
  4 000 chars, path 4 096 chars) so a single request cannot exhaust memory.
- **No secret leakage** — the OpenAI API key is never returned by any endpoint;
  backend errors are logged server-side and surface to clients as a generic
  `502` without internal URLs or config hints. Tests lock these in.
- **Backend timeouts** — both the Ollama and OpenAI calls use request timeouts.

### Authentication

Protected endpoints (`/query`, `/ingest/text`, `/ingest/path`, `/reset`) support
opt-in **Bearer API-key** authentication for server-to-server use. `/health`
stays open for liveness probes.

- Set one or more keys via `RAG_API_KEYS` (comma-separated for rotation or
  multiple consumers). Generate a strong key: `openssl rand -hex 32`.
- When `RAG_API_KEYS` is **empty, auth is disabled** — convenient for local/CLI
  use. The server logs a warning at startup in that case.
- Requests must send `Authorization: Bearer <key>`. Missing/malformed → `401`;
  present but unrecognized → `403`. Keys are compared in constant time.

```bash
export RAG_API_KEYS="$(openssl rand -hex 32)"
uvicorn rag_mvp.server:app --host 127.0.0.1 --port 8000

curl -X POST localhost:8000/query \
  -H "Authorization: Bearer $RAG_API_KEYS" \
  -H 'Content-Type: application/json' \
  -d '{"question": "what is RAG?"}'
```

This is designed for a **server-to-server** caller (e.g. a Drupal backend) that
holds the key safely. From Drupal (Guzzle):

```php
$response = \Drupal::httpClient()->post('http://rag-host:8000/query', [
  'headers' => [
    'Authorization' => 'Bearer ' . $rag_api_key, // from settings/secret store
  ],
  'json' => ['question' => $question, 'top_k' => 4],
]);
$data = json_decode((string) $response->getBody(), TRUE);
```

> The API key is only meaningful over **TLS** — otherwise it can be sniffed in
> transit. Terminate TLS at a reverse proxy (nginx/Caddy/Traefik) in front of
> the server, or keep the service on a private network. The app speaks plain
> HTTP by design; TLS is a deployment concern.
>
> Do **not** embed the key in browser/JavaScript code — anything the browser
> receives is readable by end users. Browser-facing chat should call your
> Drupal backend, which then calls this API. A browser talking directly to the
> RAG needs a different model (short-lived per-user tokens + CORS), which this
> static-key scheme intentionally does not cover.

### Known limitations (by design, for an MVP)
- **Transport security is external** — the app serves plain HTTP; put it behind
  a TLS-terminating proxy or on a trusted network.
- **Prompt injection via documents** — ingested content is placed into the LLM
  prompt as context. A malicious document could contain instructions that try
  to steer the model (e.g. "ignore previous instructions"). Retrieved text is
  data, not a trusted instruction source; treat answers over untrusted corpora
  accordingly. Full mitigation (instruction/data separation, output filtering)
  is out of scope for this MVP.
- **Dependency advisories** — run an audit periodically:
  ```bash
  uv pip install pip-audit && pip-audit
  ```
  As of writing, `pip-audit` reports advisories against `chromadb` that all
  concern its **client/server deployment mode** (the `/api/v2/...` HTTP API,
  RBAC providers, and multi-tenant isolation). This project uses ChromaDB in
  **embedded `PersistentClient` mode** — there is no Chroma server, HTTP API,
  tenant, or auth layer — so that attack surface is not exposed here. No fixed
  release was available upstream at the time of this pass; re-check with
  `pip-audit` and upgrade when one ships.

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
