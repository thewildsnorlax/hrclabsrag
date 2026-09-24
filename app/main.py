"""FastAPI application entry point.

Run with: uvicorn app.main:create_app --factory
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.documents import DocumentService
from app.ingestion import IngestionError
from app.sessions import Session, SessionStore
from app.store import Embedder, SentenceTransformerEmbedder, VectorStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PURGE_INTERVAL_SECONDS = 3600
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(settings: Optional[Settings] = None, embedder: Optional[Embedder] = None) -> FastAPI:
    settings = settings or get_settings()
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    sessions = SessionStore(
        settings.data_dir / "sessions.db", ttl_seconds=settings.session_ttl_hours * 3600
    )
    store = VectorStore(settings.data_dir / "chroma", embedder)
    documents = DocumentService(settings, sessions, store)

    def purge_expired_sessions() -> List[str]:
        expired = sessions.purge_expired()
        for session_id in expired:
            documents.forget_session(session_id)
        if expired:
            logger.info("Purged %d expired session(s)", len(expired))
        return expired

    async def purge_periodically() -> None:
        interval = min(PURGE_INTERVAL_SECONDS, sessions.ttl_seconds)
        while True:
            await asyncio.sleep(interval)
            await asyncio.to_thread(purge_expired_sessions)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        purge_expired_sessions()
        tasks = [asyncio.create_task(purge_periodically())]
        load = getattr(embedder, "load", None)
        if load is not None:  # warm the model in the background so the first upload is fast
            tasks.append(asyncio.create_task(asyncio.to_thread(load)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            sessions.close()

    app = FastAPI(title="RAG Generator", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions
    app.state.store = store
    app.state.purge_expired_sessions = purge_expired_sessions

    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(_: Request, exc: IngestionError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "filename": exc.filename or None},
        )

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        # Reject before the multipart body is spooled to disk. Per-file limits are
        # enforced again after parsing; this only bounds the whole request.
        if request.method == "POST" and request.url.path.endswith("/documents"):
            max_body = (
                settings.max_files_per_upload * settings.max_file_size_bytes
                + MULTIPART_OVERHEAD_BYTES
            )
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > max_body:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Upload is larger than the allowed total size."},
                )
        return await call_next(request)

    def current_session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or expired")
        return session

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "llm_model": settings.llm_model,
            "llm_configured": settings.llm_configured,
            "embedding_model": settings.embedding_model,
            "limits": {
                "max_file_size_mb": settings.max_file_size_mb,
                "max_pdf_pages": settings.max_pdf_pages,
                "max_files_per_upload": settings.max_files_per_upload,
                "max_docs_per_session": settings.max_docs_per_session,
                "session_ttl_hours": settings.session_ttl_hours,
                "accepted_types": [".pdf", ".txt"],
            },
        }

    @app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
    def create_session() -> dict:
        return sessions.create().to_dict(sessions.ttl_seconds)

    @app.get("/api/sessions/{session_id}")
    def get_session(session: Session = Depends(current_session)) -> dict:
        return session.to_dict(sessions.ttl_seconds)

    @app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str) -> Response:
        if not sessions.delete(session_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found or expired")
        documents.forget_session(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/sessions/{session_id}/documents", status_code=status.HTTP_201_CREATED)
    def upload_documents(
        session: Session = Depends(current_session),
        files: List[UploadFile] = File(...),
    ) -> dict:
        # Read at most limit+1 bytes per file: enough to detect "too large" without
        # loading an arbitrarily large file into memory.
        read_limit = settings.max_file_size_bytes + 1
        uploaded = [(f.filename or "", f.file.read(read_limit)) for f in files]
        records = documents.ingest(session.id, uploaded)
        return {"documents": [r.to_dict() for r in records]}

    @app.get("/api/sessions/{session_id}/documents")
    def list_documents(session: Session = Depends(current_session)) -> dict:
        return {"documents": [r.to_dict() for r in sessions.list_documents(session.id)]}

    # Mounted last so /api routes take precedence.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
