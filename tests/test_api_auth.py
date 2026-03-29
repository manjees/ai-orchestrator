"""Tests for API Key authentication middleware."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as STC
from starlette.types import ASGIApp, Receive, Scope, Send

from orchestrator.api.app import create_api_app
from orchestrator.api.auth import APIKeyAuthMiddleware
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


class _FakeClientIP:
    """ASGI middleware that overrides the client IP in the request scope."""

    def __init__(self, app: ASGIApp, *, host: str, port: int = 12345):
        self.app = app
        self.host = host
        self.port = port

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope["client"] = (self.host, self.port)
        await self.app(scope, receive, send)


def _app_with_protected_route(settings: Settings) -> FastAPI:
    app = create_api_app(settings)

    @app.get("/api/protected")
    async def protected():
        return {"ok": True}

    return app


# ── A1: Valid API key in Authorization header -> 200 ─────────────────────────


def test_valid_api_key_passes():
    app = _app_with_protected_route(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    resp = client.get("/api/protected", headers={"Authorization": "Bearer secret123"})
    assert resp.status_code == 200


# ── A2: Invalid API key -> 401 ───────────────────────────────────────────────


def test_invalid_api_key_returns_401():
    app = _app_with_protected_route(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    resp = client.get("/api/protected", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid API key"


# ── A3: Missing Authorization header, non-localhost -> 403 ───────────────────


def test_missing_auth_non_localhost_returns_403():
    app = _app_with_protected_route(_make_settings(dashboard_api_key=""))
    wrapped = _FakeClientIP(app, host="192.168.1.100")
    client = STC(wrapped)
    resp = client.get("/api/protected")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Forbidden: localhost only"


# ── A4: Missing Authorization header, localhost -> 200 ───────────────────────


def test_missing_auth_localhost_allowed():
    app = _app_with_protected_route(_make_settings(dashboard_api_key=""))
    wrapped = _FakeClientIP(app, host="127.0.0.1")
    client = STC(wrapped)
    resp = client.get("/api/protected")
    assert resp.status_code == 200


# ── A5: Empty DASHBOARD_API_KEY + IPv6 localhost -> 200 ──────────────────────


def test_empty_api_key_ipv6_localhost_allowed():
    app = _app_with_protected_route(_make_settings(dashboard_api_key=""))
    wrapped = _FakeClientIP(app, host="::1")
    client = STC(wrapped)
    resp = client.get("/api/protected")
    assert resp.status_code == 200


# ── A5b: IPv6 scoped localhost (::1%lo0) -> 200 ─────────────────────────────


def test_empty_api_key_ipv6_scoped_localhost_allowed():
    app = _app_with_protected_route(_make_settings(dashboard_api_key=""))
    wrapped = _FakeClientIP(app, host="::1%lo0")
    client = STC(wrapped)
    resp = client.get("/api/protected")
    assert resp.status_code == 200


# ── A6: Empty DASHBOARD_API_KEY + non-localhost -> 403 ───────────────────────


def test_empty_api_key_non_localhost_forbidden():
    app = _app_with_protected_route(_make_settings(dashboard_api_key=""))
    wrapped = _FakeClientIP(app, host="10.0.0.1")
    client = STC(wrapped)
    resp = client.get("/api/protected")
    assert resp.status_code == 403


# ── A7: Bearer prefix is required ────────────────────────────────────────────


def test_bearer_prefix_required():
    app = _app_with_protected_route(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    resp = client.get("/api/protected", headers={"Authorization": "secret123"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid API key"


# ── A8: Health endpoint is exempt from auth ──────────────────────────────────


def test_health_exempt_from_auth():
    app = create_api_app(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200


# ── A9: Failed auth attempts are logged ──────────────────────────────────────


def test_failed_auth_logs_warning(caplog):
    app = _app_with_protected_route(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    with caplog.at_level(logging.WARNING, logger="orchestrator.api.auth"):
        client.get("/api/protected", headers={"Authorization": "Bearer wrong"})
    assert any("auth" in r.message.lower() for r in caplog.records)


def test_failed_localhost_check_logs_warning(caplog):
    app = _app_with_protected_route(_make_settings(dashboard_api_key=""))
    wrapped = _FakeClientIP(app, host="10.0.0.1")
    client = STC(wrapped)
    with caplog.at_level(logging.WARNING, logger="orchestrator.api.auth"):
        client.get("/api/protected")
    assert any("forbidden" in r.message.lower() or "denied" in r.message.lower() for r in caplog.records)


# ── A10: Constant-time comparison (timing-safe) ─────────────────────────────


def test_auth_uses_constant_time_comparison():
    """Verify the middleware uses hmac.compare_digest (not ==) for key comparison."""
    import inspect
    source = inspect.getsource(APIKeyAuthMiddleware.dispatch)
    assert "compare_digest" in source
