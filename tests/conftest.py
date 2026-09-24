import hashlib
import math
import re
from typing import List, Sequence

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: texts sharing words are similar."""

    DIM = 256

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def make_settings(tmp_path):
    def _make(**overrides) -> Settings:
        overrides.setdefault("data_dir", tmp_path / "data")
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def make_client(make_settings):
    clients = []

    def _make(embedder=None, **overrides) -> TestClient:
        app = create_app(make_settings(**overrides), embedder=embedder or FakeEmbedder())
        client = TestClient(app)
        client.__enter__()  # run lifespan (startup purge, background task)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
