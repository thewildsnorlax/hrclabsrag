"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = Field(1024, gt=0)

    # Embeddings / storage
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = Path("./data")

    # Ingestion limits
    max_file_size_mb: int = Field(10, gt=0)
    max_pdf_pages: int = Field(200, gt=0)
    max_files_per_upload: int = Field(5, gt=0)
    max_docs_per_session: int = Field(20, gt=0)

    # Sessions / chat
    session_ttl_hours: int = Field(24, gt=0)
    max_history_turns: int = Field(6, ge=0)

    # Retrieval
    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(150, ge=0)
    top_k: int = Field(5, gt=0)

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        return bool(self.anthropic_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
