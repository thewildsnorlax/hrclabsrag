"""Turn raw PDF / TXT bytes into page-level text."""

import io
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.errors import TooManyPages, UnreadableDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Page:
    number: Optional[int]  # 1-based for PDFs; None for plain text
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    file_type: str
    pages: List[Page]

    @property
    def page_count(self) -> Optional[int]:
        return len(self.pages) if self.file_type == "pdf" else None


def load_document(filename: str, file_type: str, data: bytes, max_pdf_pages: int) -> LoadedDocument:
    if file_type == "pdf":
        pages = _load_pdf(filename, data, max_pdf_pages)
    elif file_type == "txt":
        pages = _load_txt(filename, data)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    if not any(p.text for p in pages):
        hint = " It may be a scanned PDF; OCR is not supported." if file_type == "pdf" else ""
        raise UnreadableDocument(f"No extractable text found.{hint}", filename)
    return LoadedDocument(filename=filename, file_type=file_type, pages=pages)


def _load_pdf(filename: str, data: bytes, max_pages: int) -> List[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnreadableDocument("PDF is password-protected.", filename)
        page_count = len(reader.pages)
    except UnreadableDocument:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise UnreadableDocument(f"Could not read PDF: {exc}", filename) from exc

    if page_count > max_pages:
        raise TooManyPages(
            f"PDF has {page_count} pages; the limit is {max_pages}.", filename
        )

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pypdf can fail on individual malformed pages
            logger.warning("Failed to extract text from %s page %d", filename, i, exc_info=True)
            text = ""
        pages.append(Page(number=i, text=normalize_text(text)))
    return pages


def _load_txt(filename: str, data: bytes) -> List[Page]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")  # never fails; best effort for legacy encodings
    return [Page(number=None, text=normalize_text(text))]


_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MANY_SPACES.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
