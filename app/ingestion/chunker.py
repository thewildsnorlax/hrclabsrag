"""Recursive character chunking that keeps page numbers for citations.

Text is split on the coarsest separator available (paragraphs, then lines,
sentences, words, characters) until every piece fits in `chunk_size`, then
pieces are greedily merged into chunks with `chunk_overlap` characters of
trailing context carried into the next chunk. Chunks never span pages, so
each chunk cites exactly one page.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.ingestion.loaders import LoadedDocument

SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class Chunk:
    text: str
    page: Optional[int]
    index: int  # position within the document


def chunk_document(doc: LoadedDocument, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page in doc.pages:
        for text in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text.strip():
        return []
    return _merge(_split(text, chunk_size, SEPARATORS), chunk_size, chunk_overlap)


def _split(text: str, size: int, separators: Sequence[str]) -> List[str]:
    """Break text into pieces no longer than `size`, keeping separators attached."""
    if len(text) <= size:
        return [text]
    sep_index = next(i for i, s in enumerate(separators) if s == "" or s in text)
    sep = separators[sep_index]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    pieces: List[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep
        if not part:
            continue
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_split(part, size, separators[sep_index + 1 :]))
    return pieces


def _merge(pieces: List[str], size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    window: List[str] = []
    window_len = 0
    for piece in pieces:
        if window and window_len + len(piece) > size:
            chunks.append("".join(window))
            # Keep only the tail of the window (<= overlap chars) as context for the next chunk.
            while window and (window_len > overlap or window_len + len(piece) > size):
                window_len -= len(window.pop(0))
        window.append(piece)
        window_len += len(piece)
    if window:
        chunks.append("".join(window))
    return [c.strip() for c in chunks if c.strip()]
