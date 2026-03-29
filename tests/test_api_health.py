"""Tests for the health endpoint."""

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture()
def client():
    app = create_api_app(_make_settings())
    return TestClient(app)


# ── B1: GET /api/health returns 200 with {"status": "ok"} ───────────────────


def test_health_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── B2: Response content-type is application/json ────────────────────────────


def test_health_content_type(client):
    resp = client.get("/api/health")
    assert "application/json" in resp.headers["content-type"]
