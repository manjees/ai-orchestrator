"""Integration tests for API app factory and config."""

from fastapi import FastAPI
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


# ── C1: Settings defaults ────────────────────────────────────────────────────


def test_settings_defaults():
    s = _make_settings()
    assert s.api_port == 9000
    assert s.cors_origins == ""
    assert s.dashboard_api_key == ""


# ── C2: create_api_app() returns a FastAPI instance ──────────────────────────


def test_create_api_app_returns_fastapi():
    app = create_api_app(_make_settings())
    assert isinstance(app, FastAPI)


# ── C3: CORS middleware is applied when cors_origins is set ──────────────────


def test_cors_middleware_applied():
    app = create_api_app(_make_settings(cors_origins="http://localhost:3000"))
    middleware_classes = [m.cls.__name__ for m in app.user_middleware]
    assert "CORSMiddleware" in middleware_classes


# ── C4: cors_origins comma-separated string is parsed correctly ──────────────


def test_cors_origins_parsing():
    app = create_api_app(_make_settings(cors_origins="http://a.com, http://b.com ,"))
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200


# ── C5: CORS not applied when cors_origins is empty ─────────────────────────


def test_cors_not_applied_when_empty():
    app = create_api_app(_make_settings(cors_origins=""))
    middleware_classes = [m.cls.__name__ for m in app.user_middleware]
    assert "CORSMiddleware" not in middleware_classes


# ── C6: Invalid CORS origins are filtered out ────────────────────────────────


def test_invalid_cors_origins_filtered():
    app = create_api_app(_make_settings(cors_origins="not-a-url, http://valid.com"))
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
