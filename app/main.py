"""FastAPI application entry point."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="RAG Generator", version="0.1.0")
    app.state.settings = settings

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
                "accepted_types": [".pdf", ".txt"],
            },
        }

    # Mounted last so /api routes take precedence.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
