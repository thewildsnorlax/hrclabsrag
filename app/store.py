"""Embeddings and the per-session vector store (ChromaDB, persisted to disk)."""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.ingestion import Chunk

logger = logging.getLogger(__name__)

# BGE v1.5 models retrieve better when short queries carry this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = 32


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class SentenceTransformerEmbedder:
    """Local embedding model, loaded lazily on first use (thread-safe)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        self.load()
        vectors = self._model.encode(
            list(texts), batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        self.load()
        return self._model.encode(self._query_prefix + text, normalize_embeddings=True).tolist()


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    document_id: str
    filename: str
    page: Optional[int]
    chunk_index: int
    score: float  # cosine similarity, higher is more relevant


class VectorStore:
    """One Chroma collection per session, so retrieval can never cross sessions."""

    def __init__(self, path: Path, embedder: Embedder):
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._embedder = embedder

    @staticmethod
    def _collection_name(session_id: str) -> str:
        return f"session_{session_id}"

    def _collection(self, session_id: str):
        return self._client.get_or_create_collection(
            self._collection_name(session_id),
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def add_chunks(self, session_id: str, document_id: str, filename: str, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        metadatas = []
        for c in chunks:
            meta = {"document_id": document_id, "filename": filename, "chunk_index": c.index}
            if c.page is not None:  # Chroma metadata values cannot be None
                meta["page"] = c.page
            metadatas.append(meta)
        self._collection(session_id).add(
            ids=[f"{document_id}:{c.index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )

    def delete_document(self, session_id: str, document_id: str) -> None:
        self._collection(session_id).delete(where={"document_id": document_id})

    def search(self, session_id: str, query: str, k: int) -> List[RetrievedChunk]:
        collection = self._collection(session_id)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                text=text,
                document_id=meta["document_id"],
                filename=meta["filename"],
                page=meta.get("page"),
                chunk_index=meta["chunk_index"],
                score=1.0 - distance,
            )
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def count(self, session_id: str) -> int:
        return self._collection(session_id).count()

    def delete_session(self, session_id: str) -> None:
        try:
            self._client.delete_collection(self._collection_name(session_id))
        except NotFoundError:
            pass
