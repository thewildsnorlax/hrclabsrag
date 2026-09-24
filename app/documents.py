"""Synchronous, all-or-nothing ingestion of an upload batch into a session."""

import threading
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from app.config import Settings
from app.ingestion import DuplicateDocument, ProcessedDocument, process_file, validate_batch
from app.sessions import DocumentRecord, SessionStore
from app.store import VectorStore

UploadedFile = Tuple[str, bytes]  # (filename, content)


class DocumentService:
    def __init__(self, settings: Settings, sessions: SessionStore, store: VectorStore):
        self._settings = settings
        self._sessions = sessions
        self._store = store
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[session_id]

    def ingest(self, session_id: str, files: Sequence[UploadedFile]) -> List[DocumentRecord]:
        """Validate and parse every file first; index only if all of them are accepted.

        The per-session lock serialises concurrent uploads so the document
        limit and duplicate checks cannot be raced.
        """
        with self._session_lock(session_id):
            existing = self._sessions.list_documents(session_id)
            validate_batch(len(files), len(existing), self._settings)

            seen_hashes = {d.sha256: d.filename for d in existing}
            processed: List[ProcessedDocument] = []
            for filename, data in files:
                doc = process_file(filename, data, self._settings)
                name = doc.document.filename
                if doc.sha256 in seen_hashes:
                    raise DuplicateDocument(
                        f"Same content as '{seen_hashes[doc.sha256]}', which is already in this session.",
                        name,
                    )
                seen_hashes[doc.sha256] = name
                processed.append(doc)

            added: List[DocumentRecord] = []
            try:
                for doc in processed:
                    record = self._sessions.add_document(
                        session_id=session_id,
                        filename=doc.document.filename,
                        file_type=doc.document.file_type,
                        page_count=doc.document.page_count,
                        chunk_count=len(doc.chunks),
                        size_bytes=doc.size_bytes,
                        sha256=doc.sha256,
                    )
                    added.append(record)
                    self._store.add_chunks(session_id, record.id, record.filename, doc.chunks)
            except Exception:
                for record in added:  # roll back so the batch is all-or-nothing
                    self._store.delete_document(session_id, record.id)
                    self._sessions.delete_document(record.id)
                raise
            return added

    def forget_session(self, session_id: str) -> None:
        self._store.delete_session(session_id)
        with self._locks_guard:
            self._locks.pop(session_id, None)
