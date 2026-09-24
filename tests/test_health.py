import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app


def make_client(**overrides) -> TestClient:
    settings = Settings(_env_file=None, **overrides)
    return TestClient(create_app(settings))


def test_health_reports_config_without_leaking_key():
    client = make_client(anthropic_api_key="sk-secret", llm_model="claude-haiku-4-5")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_model"] == "claude-haiku-4-5"
    assert body["llm_configured"] is True
    assert "sk-secret" not in resp.text


def test_health_flags_missing_api_key():
    body = make_client(anthropic_api_key="").get("/api/health").json()
    assert body["llm_configured"] is False


def test_default_model_is_sonnet():
    assert Settings(_env_file=None).llm_model == "claude-sonnet-5"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_root_serves_ui():
    resp = make_client().get("/")
    assert resp.status_code == 200
    assert "RAG Generator" in resp.text
