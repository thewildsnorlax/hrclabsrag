"""FastAPI application entry point.

Run with: uvicorn app.main:create_app --factory
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.sessions import Session, SessionStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
PURGE_INTERVAL_SECONDS = 3600


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    sessions = SessionStore(
        settings.data_dir / "sessions.db", ttl_seconds=settings.session_ttl_hours * 3600
    )

    def remove_session_data(session_id: str) -> None:
        """Delete everything a session owns besides its registry row (documents, vectors)."""
        # Wired to the vector store in milestone 4.

    def purge_expired_sessions() -> List[str]:
        expired = sessions.purge_expired()
        for session_id in expired:
            remove_session_data(session_id)
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
        task = asyncio.create_task(purge_periodically())
        try:
            yield
        finally:
            task.cancel()
            sessions.close()

    app = FastAPI(title="RAG Generator", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions
    app.state.purge_expired_sessions = purge_expired_sessions

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
        remove_session_data(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Mounted last so /api routes take precedence.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
