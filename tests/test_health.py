import pytest
from pydantic import ValidationError

from app.config import Settings


def test_health_reports_config_without_leaking_key(make_client):
    client = make_client(anthropic_api_key="sk-secret", llm_model="claude-haiku-4-5")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_model"] == "claude-haiku-4-5"
    assert body["llm_configured"] is True
    assert "sk-secret" not in resp.text


def test_health_flags_missing_api_key(make_client):
    body = make_client(anthropic_api_key="").get("/api/health").json()
    assert body["llm_configured"] is False


def test_default_model_is_sonnet():
    assert Settings(_env_file=None).llm_model == "claude-sonnet-5"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_root_serves_ui(make_client):
    resp = make_client().get("/")
    assert resp.status_code == 200
    assert "RAG Generator" in resp.text
