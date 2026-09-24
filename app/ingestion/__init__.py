"""Ingestion pipeline: validate -> load -> chunk."""

from dataclasses import dataclass
from typing import List

from app.config import Settings
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.errors import IngestionError
from app.ingestion.loaders import LoadedDocument, load_document
from app.ingestion.validation import clean_filename, validate_batch, validate_file

__all__ = [
    "Chunk",
    "IngestionError",
    "ProcessedDocument",
    "process_file",
    "validate_batch",
]


@dataclass(frozen=True)
class ProcessedDocument:
    document: LoadedDocument
    chunks: List[Chunk]
    size_bytes: int


def process_file(filename: str, data: bytes, settings: Settings) -> ProcessedDocument:
    """Validate, parse and chunk one uploaded file. Raises IngestionError on rejection."""
    filename = clean_filename(filename)
    file_type = validate_file(filename, data, settings)
    document = load_document(filename, file_type, data, settings.max_pdf_pages)
    chunks = chunk_document(document, settings.chunk_size, settings.chunk_overlap)
    return ProcessedDocument(document=document, chunks=chunks, size_bytes=len(data))
