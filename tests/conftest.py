import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def make_settings(tmp_path):
    def _make(**overrides) -> Settings:
        overrides.setdefault("data_dir", tmp_path / "data")
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def make_client(make_settings):
    clients = []

    def _make(**overrides) -> TestClient:
        client = TestClient(create_app(make_settings(**overrides)))
        client.__enter__()  # run lifespan (startup purge, background task)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
