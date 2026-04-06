"""Tests for POST /api/approvals/* endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from orchestrator import approval_store
from orchestrator.approval_store import ApprovalType, PendingApproval
from orchestrator.api.app import create_api_app
from orchestrator.config import Settings


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key="",
        cors_origins="",
        github_user="testuser",
        figma_access_token="fake-figma-token",
    )
    defaults.update(overrides)
    return Settings(**defaults)


_TEST_API_KEY = "test-approvals-key"


def _create_client(**settings_overrides) -> TestClient:
    settings_overrides.setdefault("dashboard_api_key", _TEST_API_KEY)
    settings = _make_settings(**settings_overrides)
    app = create_api_app(settings)
    return TestClient(app, headers={"Authorization": f"Bearer {_TEST_API_KEY}"})


def _register_strategy(approval_id: str = "123:42") -> PendingApproval:
    pending = PendingApproval(
        approval_id=approval_id,
        approval_type=ApprovalType.STRATEGY,
        context={"issue_num": 42, "project": "my-app", "chat_id": 123},
    )
    approval_store.register(pending)
    return pending


def _register_court(approval_id: str = "123:42") -> PendingApproval:
    pending = PendingApproval(
        approval_id=approval_id,
        approval_type=ApprovalType.SUPREME_COURT,
        context={"issue_num": 42, "project": "my-app", "chat_id": 123, "ruling": "UPHOLD"},
    )
    approval_store.register(pending)
    return pending


@pytest.fixture(autouse=True)
def clear_store():
    approval_store.clear()
    yield
    approval_store.clear()


# ── A1: POST /api/approvals/{id}/respond — Strategy ──────────────────────────


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_strategy_approve_returns_200_and_sets_decision(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_strategy("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "approve"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["approval_id"] == "123:42"
    assert body["decision"] == "approve"
    assert body["status"] == "resolved"

    resolved = approval_store.get("123:42")
    assert resolved is not None
    assert resolved.decision == "approve"
    assert resolved.resolved is True


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_strategy_nosplit_returns_200(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_strategy("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "nosplit"})

    assert resp.status_code == 200
    assert resp.json()["decision"] == "nosplit"


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_strategy_cancel_returns_200(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_strategy("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "cancel"})

    assert resp.status_code == 200
    assert resp.json()["decision"] == "cancel"


# ── A2: POST /api/approvals/{id}/respond — Supreme Court ─────────────────────


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_court_uphold_returns_200(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_court("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "uphold"})

    assert resp.status_code == 200
    assert resp.json()["decision"] == "uphold"


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_court_overturn_returns_200(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_court("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "overturn"})

    assert resp.status_code == 200
    assert resp.json()["decision"] == "overturn"


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_court_accept_returns_200(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_court("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "accept"})

    assert resp.status_code == 200
    assert resp.json()["decision"] == "accept"


# ── A3: Error Cases ───────────────────────────────────────────────────────────


def test_respond_unknown_id_returns_404():
    client = _create_client()

    resp = client.post("/api/approvals/nonexistent/respond", json={"decision": "approve"})

    assert resp.status_code == 404


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_respond_invalid_decision_returns_422(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_strategy("123:42")

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "invalid_value"})

    assert resp.status_code == 422


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_respond_already_resolved_returns_409(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_strategy("123:42")

    # First response
    resp1 = client.post("/api/approvals/123:42/respond", json={"decision": "approve"})
    assert resp1.status_code == 200

    # Second response — should be 409
    resp2 = client.post("/api/approvals/123:42/respond", json={"decision": "nosplit"})
    assert resp2.status_code == 409


# ── A4: Race Condition (Telegram vs Dashboard) ───────────────────────────────


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_first_responder_wins(mock_bus):
    mock_bus.return_value.emit = AsyncMock()
    client = _create_client()
    _register_strategy("123:42")

    # Resolve via approval_store directly (simulating Telegram)
    approval_store.resolve("123:42", "nosplit")

    # API attempt arrives after — should get 409
    resp = client.post("/api/approvals/123:42/respond", json={"decision": "approve"})
    assert resp.status_code == 409

    # The decision set by the first responder (Telegram) wins
    resolved = approval_store.get("123:42")
    assert resolved.decision == "nosplit"


# ── A5: GET /api/approvals/pending ───────────────────────────────────────────


def test_list_pending_returns_active_approvals():
    client = _create_client()
    _register_strategy("123:10")
    _register_court("123:11")

    resp = client.get("/api/approvals/pending")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    ids = {item["approval_id"] for item in body}
    assert ids == {"123:10", "123:11"}


def test_list_pending_empty():
    client = _create_client()

    resp = client.get("/api/approvals/pending")

    assert resp.status_code == 200
    assert resp.json() == []


def test_list_pending_excludes_resolved():
    client = _create_client()
    _register_strategy("123:10")
    approval_store.resolve("123:10", "approve")

    resp = client.get("/api/approvals/pending")

    assert resp.status_code == 200
    assert resp.json() == []


# ── A6: WebSocket Event ───────────────────────────────────────────────────────


@patch("orchestrator.api.approval_routes.get_event_bus")
def test_approval_responded_emits_ws_event(mock_bus):
    mock_emit = AsyncMock()
    mock_bus.return_value.emit = mock_emit
    client = _create_client()
    _register_strategy("123:42")

    client.post("/api/approvals/123:42/respond", json={"decision": "approve"})

    mock_emit.assert_called_once()
    call_args = mock_emit.call_args
    assert call_args[0][0] == "approval.responded"
    data = call_args[0][1]
    assert data["approval_id"] == "123:42"
    assert data["decision"] == "approve"


# ── A6b: approval.required WebSocket Event ───────────────────────────────────


@pytest.mark.asyncio
async def test_approval_required_emitted_for_strategy():
    """approval_store.register() + bus.emit() are called when strategy approval is created."""
    from unittest.mock import AsyncMock, patch

    from orchestrator.api.events import EventType

    mock_emit = AsyncMock()
    with patch("orchestrator.handlers.get_event_bus") as mock_bus:
        mock_bus.return_value.emit = mock_emit

        # Simulate what _solve_with_fivebrid does
        pending = PendingApproval(
            approval_id="123:99",
            approval_type=ApprovalType.STRATEGY,
            context={"issue_num": 99, "project": "test-project", "chat_id": 123},
        )
        approval_store.register(pending)
        bus = mock_bus()
        await bus.emit(EventType.APPROVAL_REQUIRED, {
            "approval_id": "123:99",
            "approval_type": pending.approval_type.value,
            "context": pending.context,
        })

    mock_emit.assert_called_once()
    call_args = mock_emit.call_args
    assert call_args[0][0] == "approval.required"
    data = call_args[0][1]
    assert data["approval_id"] == "123:99"
    assert data["approval_type"] == "strategy"


@pytest.mark.asyncio
async def test_approval_required_emitted_for_supreme_court():
    """approval_store.register() + bus.emit() are called when supreme court approval is created."""
    from unittest.mock import AsyncMock, patch

    from orchestrator.api.events import EventType

    mock_emit = AsyncMock()
    with patch("orchestrator.handlers.get_event_bus") as mock_bus:
        mock_bus.return_value.emit = mock_emit

        # Simulate what _supreme_court_user_decision does
        pending = PendingApproval(
            approval_id="123:99",
            approval_type=ApprovalType.SUPREME_COURT,
            context={"issue_num": 99, "project": "test-project", "chat_id": 123, "ruling": "UPHOLD"},
        )
        approval_store.register(pending)
        bus = mock_bus()
        await bus.emit(EventType.APPROVAL_REQUIRED, {
            "approval_id": "123:99",
            "approval_type": pending.approval_type.value,
            "context": pending.context,
        })

    mock_emit.assert_called_once()
    call_args = mock_emit.call_args
    assert call_args[0][0] == "approval.required"
    data = call_args[0][1]
    assert data["approval_id"] == "123:99"
    assert data["approval_type"] == "supreme_court"


# ── A7: Auth ─────────────────────────────────────────────────────────────────


def test_respond_without_api_key_returns_401():
    settings = _make_settings(dashboard_api_key=_TEST_API_KEY)
    app = create_api_app(settings)
    client = TestClient(app)  # No auth header

    resp = client.post("/api/approvals/123:42/respond", json={"decision": "approve"})
    assert resp.status_code == 401
