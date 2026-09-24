"""Upload limit checks, run before any parsing so oversized input is rejected cheaply."""

from pathlib import PurePath

from app.config import Settings
from app.ingestion.errors import (
    FileTooLarge,
    SessionDocumentLimit,
    TooManyFiles,
    UnsupportedFileType,
)

SUPPORTED_TYPES = {".pdf": "pdf", ".txt": "txt"}
MAX_FILENAME_LENGTH = 255


def validate_batch(file_count: int, existing_doc_count: int, settings: Settings) -> None:
    if file_count == 0:
        raise TooManyFiles("No files were uploaded.")
    if file_count > settings.max_files_per_upload:
        raise TooManyFiles(
            f"Too many files in one upload ({file_count}); "
            f"the limit is {settings.max_files_per_upload}."
        )
    if existing_doc_count + file_count > settings.max_docs_per_session:
        remaining = max(settings.max_docs_per_session - existing_doc_count, 0)
        raise SessionDocumentLimit(
            f"This session can hold at most {settings.max_docs_per_session} documents; "
            f"{remaining} more can be added."
        )


def clean_filename(filename: str) -> str:
    """Strip any client-supplied directory components ("../../x.pdf" -> "x.pdf")."""
    name = PurePath((filename or "").replace("\\", "/")).name.strip()
    return name[:MAX_FILENAME_LENGTH] or "untitled"


def validate_file(filename: str, data: bytes, settings: Settings) -> str:
    """Check size and type of one file; return its type ("pdf" or "txt")."""
    suffix = PurePath(filename).suffix.lower()
    file_type = SUPPORTED_TYPES.get(suffix)
    if file_type is None:
        raise UnsupportedFileType(
            f"Unsupported file type '{suffix or filename}'. Supported types: .pdf, .txt.",
            filename,
        )
    if len(data) == 0:
        raise UnsupportedFileType("File is empty.", filename)
    if len(data) > settings.max_file_size_bytes:
        # The API reads at most limit+1 bytes, so the true size is unknown here.
        raise FileTooLarge(
            f"File exceeds the size limit; the limit is {settings.max_file_size_mb} MB.",
            filename,
        )
    # Check content matches the extension, so e.g. a renamed binary isn't accepted.
    if file_type == "pdf" and b"%PDF-" not in data[:1024]:
        raise UnsupportedFileType("File has a .pdf extension but is not a PDF.", filename)
    if file_type == "txt" and b"\x00" in data[:8192]:
        raise UnsupportedFileType("File has a .txt extension but looks binary.", filename)
    return file_type
