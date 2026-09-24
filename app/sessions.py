"""Session and document registry backed by SQLite, so both survive server restarts.

A session is the unit of isolation: it owns one document set and one chat.
Sessions expire after `ttl` of inactivity; every successful lookup refreshes
the activity timestamp. Chunk vectors live in the vector store; this registry
holds the per-document metadata used for listing and limit checks.
"""

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

Clock = Callable[[], float]


@dataclass(frozen=True)
class Session:
    id: str
    created_at: float
    last_active_at: float

    def to_dict(self, ttl_seconds: float) -> dict:
        return {
            "session_id": self.id,
            "created_at": _iso(self.created_at),
            "last_active_at": _iso(self.last_active_at),
            "expires_at": _iso(self.last_active_at + ttl_seconds),
        }


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    session_id: str
    filename: str
    file_type: str
    page_count: Optional[int]
    chunk_count: int
    size_bytes: int
    sha256: str
    created_at: float

    def to_dict(self) -> dict:
        return {
            "document_id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "size_bytes": self.size_bytes,
            "created_at": _iso(self.created_at),
        }


_DOC_COLUMNS = (
    "id, session_id, filename, file_type, page_count, chunk_count, size_bytes, sha256, created_at"
)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class SessionStore:
    def __init__(self, db_path: Path, ttl_seconds: float, clock: Clock = time.time):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._clock = clock
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                       id TEXT PRIMARY KEY,
                       created_at REAL NOT NULL,
                       last_active_at REAL NOT NULL
                   )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                       id TEXT PRIMARY KEY,
                       session_id TEXT NOT NULL,
                       filename TEXT NOT NULL,
                       file_type TEXT NOT NULL,
                       page_count INTEGER,
                       chunk_count INTEGER NOT NULL,
                       size_bytes INTEGER NOT NULL,
                       sha256 TEXT NOT NULL,
                       created_at REAL NOT NULL
                   )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_session ON documents (session_id)"
            )

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def create(self) -> Session:
        now = self._clock()
        session = Session(id=uuid.uuid4().hex, created_at=now, last_active_at=now)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, created_at, last_active_at) VALUES (?, ?, ?)",
                (session.id, session.created_at, session.last_active_at),
            )
        return session

    def get(self, session_id: str, touch: bool = True) -> Optional[Session]:
        """Return the session if it exists and has not expired; refresh its activity."""
        now = self._clock()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id, created_at, last_active_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            session = Session(*row)
            if self._is_expired(session, now):
                return None  # removed (with its documents) by purge_expired
            if touch:
                self._conn.execute(
                    "UPDATE sessions SET last_active_at = ? WHERE id = ?", (now, session_id)
                )
                session = Session(session.id, session.created_at, now)
        return session

    def delete(self, session_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0

    def purge_expired(self) -> List[str]:
        """Delete expired sessions and return their ids so callers can clean up their data."""
        cutoff = self._clock() - self._ttl
        with self._lock, self._conn:
            ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM sessions WHERE last_active_at < ?", (cutoff,)
                )
            ]
            params = [(i,) for i in ids]
            self._conn.executemany("DELETE FROM sessions WHERE id = ?", params)
            self._conn.executemany("DELETE FROM documents WHERE session_id = ?", params)
        return ids

    # --- documents ---------------------------------------------------------

    def add_document(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        page_count: Optional[int],
        chunk_count: int,
        size_bytes: int,
        sha256: str,
    ) -> DocumentRecord:
        record = DocumentRecord(
            id=uuid.uuid4().hex,
            session_id=session_id,
            filename=filename,
            file_type=file_type,
            page_count=page_count,
            chunk_count=chunk_count,
            size_bytes=size_bytes,
            sha256=sha256,
            created_at=self._clock(),
        )
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO documents ({_DOC_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.session_id,
                    record.filename,
                    record.file_type,
                    record.page_count,
                    record.chunk_count,
                    record.size_bytes,
                    record.sha256,
                    record.created_at,
                ),
            )
        return record

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def list_documents(self, session_id: str) -> List[DocumentRecord]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_DOC_COLUMNS} FROM documents WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        return [DocumentRecord(*row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def _is_expired(self, session: Session, now: float) -> bool:
        return session.last_active_at < now - self._ttl
