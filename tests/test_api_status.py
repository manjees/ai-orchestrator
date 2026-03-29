"""Tests for the GET /api/status endpoint."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from orchestrator.api.app import create_api_app
from orchestrator.config import Settings


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key="",
        cors_origins="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@dataclass(frozen=True)
class _FakeSystemStatus:
    ram_total_gb: float = 16.0
    ram_used_gb: float = 8.0
    ram_percent: float = 50.0
    cpu_percent: float = 20.0
    thermal_pressure: str = "nominal"
    disk_total_gb: float = 500.0
    disk_used_gb: float = 100.0
    disk_percent: float = 20.0


@dataclass(frozen=True)
class _FakeOllamaModel:
    name: str = "llama3"
    size_gb: float = 4.7


@dataclass(frozen=True)
class _FakeTmuxSession:
    name: str = "dev"
    windows: int = 3
    created: str = "2025-01-01"


_TEST_API_KEY = "test-api-key-for-status"


@pytest.fixture()
def client():
    app = create_api_app(_make_settings(dashboard_api_key=_TEST_API_KEY))
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {_TEST_API_KEY}"
    return c


@pytest.fixture()
def _mock_deps():
    """Patch the three async dependencies with default return values."""
    with (
        patch("orchestrator.api.routes.get_system_status", new_callable=AsyncMock, return_value=_FakeSystemStatus()) as mock_sys,
        patch("orchestrator.api.routes.list_sessions", new_callable=AsyncMock, return_value=[]) as mock_sessions,
        patch("orchestrator.api.routes._get_ollama_models", new_callable=AsyncMock, return_value=[]) as mock_models,
    ):
        yield {"sys": mock_sys, "sessions": mock_sessions, "models": mock_models}


# -- Test 1: Status returns 200 with valid schema --


def test_status_returns_200_with_valid_schema(client, _mock_deps):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "ram_total_gb", "ram_used_gb", "ram_percent", "cpu_percent",
        "thermal_pressure", "disk_total_gb", "disk_used_gb", "disk_percent",
        "ollama_models", "tmux_sessions",
    ):
        assert key in data, f"Missing key: {key}"
    assert isinstance(data["ram_total_gb"], float)
    assert isinstance(data["thermal_pressure"], str)
    assert isinstance(data["ollama_models"], list)
    assert isinstance(data["tmux_sessions"], list)


# -- Test 2: Status contains ollama models --


@pytest.mark.parametrize(
    "model_name,model_size",
    [("llama3", 4.7), ("qwen2.5-coder", 8.0), ("gemma2", 2.6)],
)
def test_status_contains_ollama_models(client, _mock_deps, model_name, model_size):
    _mock_deps["models"].return_value = [_FakeOllamaModel(name=model_name, size_gb=model_size)]
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["ollama_models"]) == 1
    assert data["ollama_models"][0]["name"] == model_name
    assert data["ollama_models"][0]["size_gb"] == model_size


# -- Test 3: Status contains tmux sessions --


def test_status_contains_tmux_sessions(client, _mock_deps):
    _mock_deps["sessions"].return_value = [_FakeTmuxSession(name="dev", windows=3, created="2025-01-01")]
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tmux_sessions"]) == 1
    assert data["tmux_sessions"][0]["name"] == "dev"
    assert data["tmux_sessions"][0]["windows"] == 3
    assert data["tmux_sessions"][0]["created"] == "2025-01-01"


# -- Test 4: Graceful when services unavailable --


def test_status_graceful_when_services_unavailable(client, _mock_deps):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ollama_models"] == []
    assert data["tmux_sessions"] == []


# -- Test 5: Secret masking applied --


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-abc123XYZ0123456789",   # Anthropic key
        "sk-proj-abcdefghijklmnopqrstuv",  # OpenAI key
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",  # GitHub PAT
        "xoxb-123-456-abcdefgh",         # Slack bot token
    ],
)
def test_status_applies_secret_masking(client, _mock_deps, secret):
    _mock_deps["sessions"].return_value = [
        _FakeTmuxSession(name=f"dev-{secret}", windows=1, created="2025-01-01"),
    ]
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.text
    assert secret not in body
    assert "[MASKED]" in body


# -- Test 6: Auth required when api_key is set --


def test_status_requires_auth_when_api_key_set(_mock_deps):
    app = create_api_app(_make_settings(dashboard_api_key="test-key-123"))
    secured_client = TestClient(app)
    resp = secured_client.get("/api/status")
    assert resp.status_code == 401


# -- Test 7: Pydantic schema validation --


def test_status_response_model_matches_pydantic_schema():
    from orchestrator.api.schemas import StatusResponse

    valid = {
        "ram_total_gb": 16.0, "ram_used_gb": 8.0, "ram_percent": 50.0,
        "cpu_percent": 20.0, "thermal_pressure": "nominal",
        "disk_total_gb": 500.0, "disk_used_gb": 100.0, "disk_percent": 20.0,
        "ollama_models": [{"name": "llama3", "size_gb": 4.7}],
        "tmux_sessions": [{"name": "dev", "windows": 3, "created": "2025-01-01"}],
    }
    StatusResponse.model_validate(valid)

    with pytest.raises(ValidationError):
        StatusResponse.model_validate({"ram_total_gb": 16.0})  # missing required fields


# -- Test 8: Graceful degradation when ollama raises exception --


def test_status_graceful_when_ollama_raises(client):
    with (
        patch("orchestrator.api.routes.get_system_status", new_callable=AsyncMock, return_value=_FakeSystemStatus()),
        patch("orchestrator.api.routes.list_sessions", new_callable=AsyncMock, return_value=[]),
        patch("orchestrator.api.routes._get_ollama_models", new_callable=AsyncMock, side_effect=Exception("connection refused")),
    ):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ollama_models"] == []


# -- Test 9: Graceful degradation when tmux raises exception --


def test_status_graceful_when_tmux_raises(client):
    with (
        patch("orchestrator.api.routes.get_system_status", new_callable=AsyncMock, return_value=_FakeSystemStatus()),
        patch("orchestrator.api.routes.list_sessions", new_callable=AsyncMock, side_effect=Exception("tmux not found")),
        patch("orchestrator.api.routes._get_ollama_models", new_callable=AsyncMock, return_value=[]),
    ):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tmux_sessions"] == []


# -- Test 10: 500 when system_status raises (genuine server error) --


def test_status_500_when_system_monitor_raises():
    app = create_api_app(_make_settings(dashboard_api_key=_TEST_API_KEY))
    error_client = TestClient(app, raise_server_exceptions=False)
    error_client.headers["Authorization"] = f"Bearer {_TEST_API_KEY}"
    with (
        patch("orchestrator.api.routes.get_system_status", new_callable=AsyncMock, side_effect=Exception("psutil crashed")),
        patch("orchestrator.api.routes.list_sessions", new_callable=AsyncMock, return_value=[]),
        patch("orchestrator.api.routes._get_ollama_models", new_callable=AsyncMock, return_value=[]),
    ):
        resp = error_client.get("/api/status")
    assert resp.status_code == 500


# -- Test 11: Both ollama and tmux fail simultaneously --


def test_status_graceful_when_both_external_services_fail(client):
    with (
        patch("orchestrator.api.routes.get_system_status", new_callable=AsyncMock, return_value=_FakeSystemStatus()),
        patch("orchestrator.api.routes.list_sessions", new_callable=AsyncMock, side_effect=RuntimeError("tmux dead")),
        patch("orchestrator.api.routes._get_ollama_models", new_callable=AsyncMock, side_effect=ConnectionError("ollama down")),
    ):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ollama_models"] == []
    assert data["tmux_sessions"] == []
    # System status fields should still be present
    assert data["ram_total_gb"] == 16.0
