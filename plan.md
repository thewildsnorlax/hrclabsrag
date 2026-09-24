# RAG Generator — Plan

## Problem

Build a RAG generator that:
- accepts documents at runtime,
- creates a RAG application over those documents,
- lets users ask questions and receive grounded answers,
- works with different document sets without code changes.

## Decisions

| Area | Decision |
|---|---|
| Stack | Python + FastAPI |
| LLM | Claude, default `claude-sonnet-5`, configurable via `LLM_MODEL` env var |
| Embeddings | Local `sentence-transformers` model (`BAAI/bge-small-en-v1.5`), no extra API key |
| Vector store | ChromaDB (embedded, persisted to disk), one collection per session |
| Ingestion | Synchronous, within the upload request, bounded by size limits |
| Corpus model | One document set and one chat per session |
| Formats | PDF, TXT |
| UI | Single static HTML/JS page served by FastAPI |

## Scope

### Current scope
1. **Synchronous document ingestion with size caps** — an upload is parsed, chunked, embedded and indexed before the request returns. Limits keep request time bounded (see [Constraints](#constraints)).
2. **Supported formats** — PDF (text-based) and TXT.
3. **Chat per document set per session** — each session owns its own document set and its own chat. Sessions are isolated: questions in one session never retrieve documents from another.

### Future enhancements
1. **Knowledge bases** — users create and maintain named, long-lived document collections.
2. **Asynchronous ingestion** — uploads enqueue a job on a message/task queue (e.g. Celery/RQ + Redis); workers index in the background and the UI polls or subscribes for status. Removes the need for tight size caps.
3. **More document types** — HTML, DOCX, and others via pluggable loaders.
4. **Knowledge base selection in chat** — pick one or more knowledge bases to query from within a chat.

### Out of scope
OCR for scanned PDFs, authentication/user accounts, URL ingestion.

## Constraints

All limits are configurable via `.env`; defaults below.

| Constraint | Default | Rationale |
|---|---|---|
| Max file size | 10 MB (`MAX_FILE_SIZE_MB`) | Bounds synchronous ingestion time and memory |
| Max PDF pages per file | 200 (`MAX_PDF_PAGES`) | File size alone does not bound text volume; keeps embedding time within a request |
| Max files per upload request | 5 (`MAX_FILES_PER_UPLOAD`) | Bounds total work per request |
| Max documents per session | 20 (`MAX_DOCS_PER_SESSION`) | Keeps per-session index small and retrieval fast |
| Accepted types | `.pdf`, `.txt` | Checked by extension and content (PDF magic bytes / UTF-8 decodability) |
| TXT encoding | UTF-8, fallback latin-1 | Avoids rejecting common non-UTF-8 files |
| Session lifetime | Expires after 24 h idle (`SESSION_TTL_HOURS`) | Expired sessions and their collections are purged at startup and periodically |
| Chat history | Last 6 turns sent by the client with each question | Supports follow-up questions while keeping the server stateless for chat |
| Retrieval | top-k = 5 (`TOP_K`), chunk ~800 chars / 150 overlap | Fits comfortably in the prompt; tunable |

Rejections return a clear 4xx error: oversized file, too many pages/files, unsupported type, PDF with no extractable text, session document limit reached, unknown/expired session.

## Requirements

### Functional
1. **Sessions** — the UI creates a session on load; the session id is kept in the browser so a refresh resumes it. A "New session" action starts over with an empty document set and chat.
2. **Runtime upload** — upload PDF/TXT files into the current session via the web UI or REST API, within the constraints above.
3. **Automatic indexing** — each accepted upload is indexed into the session's collection; no restart or code change.
4. **Grounded Q&A** — answers are generated only from passages retrieved from the session's documents, with inline citations (`[1]`, `[2]`) mapped to file name and page. If the answer is not in the documents, the model says so rather than guessing.
5. **Document list** — view the documents indexed in the current session.
6. **Streaming** — answers stream to the UI as they are generated.

### Non-functional
- Configuration via `.env`: `ANTHROPIC_API_KEY`, `LLM_MODEL`, limits, chunking, top-k.
- Only external dependency at runtime is the Anthropic API.
- Session documents persist across server restarts until the session expires.
- Clear error when the API key is missing.

## Architecture

```
Browser (static HTML/JS, holds session id + chat history)
   │  create session / upload / ask
FastAPI ──► Ingestion (sync): validate limits → parse (pypdf / text) → chunk (keeps page no.)
   │                            → embed (bge-small-en-v1.5) → ChromaDB collection "session_<id>"
   └────► Query: embed question → top-k retrieval from session collection
                   → prompt (numbered context + recent history) → Claude (streamed via SSE)
                   → answer + citation list
```

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/sessions` | Create a session; returns `session_id` |
| `GET` | `/api/sessions/{id}` | Check a session is still valid (used to resume after a page refresh) |
| `DELETE` | `/api/sessions/{id}` | Delete a session and its documents |
| `POST` | `/api/sessions/{id}/documents` | Upload files (multipart) and index them synchronously |
| `GET` | `/api/sessions/{id}/documents` | List indexed documents |
| `POST` | `/api/sessions/{id}/query` | Ask a question (+ recent history); streams answer + citations (SSE) |
| `GET` | `/api/health` | Health / config check |

## Project layout

```
app/
  main.py            # FastAPI app, routes, static mount
  config.py          # pydantic-settings loaded from .env (model, limits, chunking)
  sessions.py        # session create/lookup/expiry
  ingestion/
    validation.py    # size / type / page / count limits
    loaders.py       # PDF / TXT parsing
    chunker.py       # chunking with page metadata
  store.py           # embeddings + Chroma wrapper (collection per session)
  rag.py             # retrieval, prompt building, Claude streaming
  static/index.html  # UI
tests/               # validation, loaders, chunker, retrieval, sessions, API (LLM mocked)
sample_docs/         # two unrelated document sets for demoing isolation between sessions
.env.example
requirements.txt
README.md
```

## Milestones

1. **Scaffold** — config (incl. limits), health endpoint, dependencies, `.env.example`.
2. **Sessions** — create/delete/lookup, expiry purge; tests.
3. **Ingestion** — validation, loaders, chunker preserving page metadata; tests for each limit.
4. **Vector store** — per-session collections: embed/upsert, list, delete, retrieve; tests incl. session isolation.
5. **RAG query** — grounded prompt, citations, "not found" behaviour, history, SSE streaming; API tests with the LLM mocked.
6. **UI** — session handling, upload with limit errors, document list, chat with streamed answers and citations.
7. **End-to-end check** — two sessions with the two sample document sets; confirm grounded answers, isolation, limit rejections, and "don't know" on off-topic questions.
8. **README** — setup, run, API usage, design decisions, constraints, future enhancements; optional Dockerfile.
9. **Submission** — commit code and export AI agent transcripts (`/export`) into `transcripts/`.
