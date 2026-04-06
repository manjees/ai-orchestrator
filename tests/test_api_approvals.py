"""Unit tests for approval API endpoints: GET /api/approvals, POST /api/approvals/{id}/respond."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orchestrator import approval_store
from orchestrator.approval_store import ApprovalType
from orchestrator.api.app import create_api_app
from orchestrator.config import Settings

_TEST_API_KEY = "test-key-for-approvals"


def _make_client() -> TestClient:
    settings = Settings(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key=_TEST_API_KEY,
        cors_origins="",
    )
    app = create_api_app(settings, projects={}, pipelines={})
    return TestClient(app, headers={"Authorization": f"Bearer {_TEST_API_KEY}"})


@pytest.fixture()
def client():
    return _make_client()


@pytest.fixture(autouse=True)
def cleanup_store():
    yield
    approval_store.clear_all()


# ── T13: POST respond with valid decision → 200 ───────────────────────────────

def test_respond_valid_decision_returns_200(client):
    approval_store.create_approval(
        "chat1:42", "strategy", ["approve", "nosplit", "cancel"]
    )
    resp = client.post(
        "/api/approvals/chat1:42/respond", json={"decision": "approve"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["approval_id"] == "chat1:42"
    assert data["status"] == "decided"
    assert data["decision"] == "approve"


# ── T14: POST respond with non-existent id → 404 ────────────────────────────

def test_respond_nonexistent_returns_404(client):
    resp = client.post(
        "/api/approvals/nonexistent/respond", json={"decision": "approve"}
    )
    assert resp.status_code == 404


# ── T15: POST respond on already-decided → 409 Conflict ─────────────────────

def test_respond_already_decided_returns_409(client):
    approval_store.create_approval("chat1:42", "strategy", ["approve", "cancel"])
    approval_store.respond("chat1:42", "approve")
    resp = client.post(
        "/api/approvals/chat1:42/respond", json={"decision": "cancel"}
    )
    assert resp.status_code == 409


# ── T16: POST respond with invalid decision → 400 ────────────────────────────

def test_respond_invalid_decision_returns_400(client):
    approval_store.create_approval("chat1:42", "strategy", ["approve", "cancel"])
    resp = client.post(
        "/api/approvals/chat1:42/respond", json={"decision": "fly_away"}
    )
    assert resp.status_code == 400


# ── T17: POST respond missing body → 422 ─────────────────────────────────────

def test_respond_missing_body_returns_422(client):
    approval_store.create_approval("chat1:42", "strategy", ["approve"])
    resp = client.post("/api/approvals/chat1:42/respond", json={})
    assert resp.status_code == 422


# ── T18: GET /api/approvals returns list of pending approvals ─────────────────

def test_list_approvals_returns_pending(client):
    approval_store.create_approval(
        "chat1:10", "strategy", ["approve", "cancel"], context={"issue_num": 10}
    )
    approval_store.create_approval(
        "chat1:11", "strategy", ["approve", "cancel"]
    )
    approval_store.respond("chat1:11", "cancel")  # decided — should not appear

    resp = client.get("/api/approvals")
    assert resp.status_code == 200
    data = resp.json()
    ids = [a["approval_id"] for a in data["approvals"]]
    assert "chat1:10" in ids
    assert "chat1:11" not in ids


# ── T19: GET /api/approvals when empty → empty list ──────────────────────────

def test_list_approvals_empty(client):
    resp = client.get("/api/approvals")
    assert resp.status_code == 200
    assert resp.json() == {"approvals": []}


# ── T20: No auth → 401 ───────────────────────────────────────────────────────

def test_respond_no_auth_returns_401():
    settings = Settings(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key=_TEST_API_KEY,
        cors_origins="",
    )
    app = create_api_app(settings, projects={}, pipelines={})
    no_auth_client = TestClient(app)  # no Authorization header
    approval_store.create_approval("chat1:42", "strategy", ["approve"])
    resp = no_auth_client.post(
        "/api/approvals/chat1:42/respond", json={"decision": "approve"}
    )
    assert resp.status_code == 401


# ── T21: GET approvals response shape ────────────────────────────────────────

def test_list_approvals_response_shape(client):
    approval_store.create_approval(
        "chat1:99", "strategy", ["approve", "nosplit", "cancel"],
        context={"issue_num": 99}
    )
    resp = client.get("/api/approvals")
    assert resp.status_code == 200
    approval = resp.json()["approvals"][0]
    assert approval["approval_id"] == "chat1:99"
    assert approval["type"] == "strategy"
    assert approval["status"] == "pending"
    assert approval["decision"] is None
    assert set(approval["decision_options"]) == {"approve", "nosplit", "cancel"}
    assert approval["context"] == {"issue_num": 99}


# ── T22: GET response serializes enum type as string ─────────────────────────

def test_list_approvals_serializes_type_as_string(client):
    approval_store.create_approval(
        "court:1:5", ApprovalType.SUPREME_COURT, ["uphold", "overturn"]
    )
    resp = client.get("/api/approvals")
    assert resp.status_code == 200
    assert resp.json()["approvals"][0]["type"] == "supreme_court"


# ── T23: POST respond works for supreme_court approval ───────────────────────

def test_respond_supreme_court_approval(client):
    approval_store.create_approval(
        "court:1:5", ApprovalType.SUPREME_COURT, ["uphold", "overturn"]
    )
    resp = client.post("/api/approvals/court:1:5/respond", json={"decision": "uphold"})
    assert resp.status_code == 200
