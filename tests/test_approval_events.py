"""Integration tests for WebSocket approval events + timeout scenarios."""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

from orchestrator import approval_store
from orchestrator.approval_store import ApprovalStatus, ApprovalType, make_approval_id
from orchestrator.api.events import EventType, WSEvent, get_event_bus, reset_event_bus


# ── Helpers ─────────────────────────────────────────────────────────────────


class FakeWS:
    """Fake WebSocket that records sent messages."""

    def __init__(self):
        self.received: list[dict] = []

    async def send_text(self, data: str) -> None:
        self.received.append(json.loads(data))


@pytest.fixture(autouse=True)
def reset_state():
    reset_event_bus()
    approval_store.clear_all()
    yield
    reset_event_bus()
    approval_store.clear_all()


# ── Group A: APPROVAL_REQUIRED WebSocket Event ───────────────────────────────


@pytest.mark.asyncio
async def test_strategy_approval_emits_required_event():
    """T1: Creating a strategy approval emits approval.required event with correct fields."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    approval_store.create_approval(
        "123:1",
        ApprovalType.STRATEGY,
        ["approve", "nosplit", "cancel"],
        context={"triage_result": {"model": "haiku"}},
    )
    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "123:1",
        "type": "strategy",
        "decision_options": ["approve", "nosplit", "cancel"],
        "context": {"triage_result": {"model": "haiku"}},
    })

    assert len(ws.received) == 1
    event = ws.received[0]
    assert event["event_type"] == "approval.required"
    assert event["data"]["approval_id"] == "123:1"
    assert event["data"]["type"] == "strategy"
    assert event["data"]["decision_options"] == ["approve", "nosplit", "cancel"]
    assert "triage_result" in event["data"]["context"]


@pytest.mark.asyncio
async def test_supreme_court_approval_emits_required_event():
    """T2: Creating a supreme_court approval emits approval.required event with type='supreme_court'."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    approval_store.create_approval(
        "court:123:5",
        ApprovalType.SUPREME_COURT,
        ["accept", "uphold", "overturn"],
        context={"ruling": "UPHOLD", "issue_num": 5},
    )
    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "court:123:5",
        "type": "supreme_court",
        "decision_options": ["accept", "uphold", "overturn"],
        "context": {"ruling": "UPHOLD", "issue_num": 5},
    })

    assert len(ws.received) == 1
    event = ws.received[0]
    assert event["event_type"] == "approval.required"
    assert event["data"]["type"] == "supreme_court"
    assert event["data"]["approval_id"] == "court:123:5"


@pytest.mark.asyncio
async def test_approval_required_payload_matches_wsevent_schema():
    """T3: approval.required event payload matches WSEvent schema."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "123:1",
        "type": "strategy",
        "decision_options": ["approve"],
        "context": {},
    })

    assert len(ws.received) == 1
    raw = ws.received[0]
    # WSEvent schema: event_type, data, timestamp
    event = WSEvent(**raw)
    assert event.event_type == EventType.APPROVAL_REQUIRED
    assert "approval_id" in event.data
    assert event.timestamp  # non-empty ISO 8601


@pytest.mark.asyncio
async def test_multiple_ws_clients_receive_approval_required():
    """T4: Multiple WS clients all receive the approval.required broadcast."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws1, ws2, ws3 = FakeWS(), FakeWS(), FakeWS()
    await bus.register(ws1)
    await bus.register(ws2)
    await bus.register(ws3)

    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "123:1",
        "type": "strategy",
        "decision_options": ["approve"],
        "context": {},
    })

    for ws in (ws1, ws2, ws3):
        assert len(ws.received) == 1
        assert ws.received[0]["event_type"] == "approval.required"


# ── Group B: APPROVAL_RESPONDED WebSocket Event ───────────────────────────────


_TEST_API_KEY = "test-key-events"


def _make_test_client():
    from fastapi.testclient import TestClient
    from orchestrator.api.app import create_api_app
    from orchestrator.config import Settings

    settings = Settings(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key=_TEST_API_KEY,
        cors_origins="",
    )
    app = create_api_app(settings)
    return TestClient(app, headers={"Authorization": f"Bearer {_TEST_API_KEY}"})


@pytest.mark.asyncio
async def test_api_respond_emits_approval_responded_event():
    """T5: API POST /api/approvals/{id}/respond emits approval.responded event."""

    approval_store.create_approval(
        "123:42", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"]
    )

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    # Use thread to post the API request (TestClient is sync)
    result = {}

    def do_request():
        client = _make_test_client()
        r = client.post("/api/approvals/123:42/respond", json={"decision": "approve"})
        result["status_code"] = r.status_code

    t = threading.Thread(target=do_request)
    t.start()
    t.join(timeout=5)

    assert result.get("status_code") == 200

    # Find the approval.responded event
    responded_events = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded_events) == 1
    event = responded_events[0]
    assert event["data"]["approval_id"] == "123:42"
    assert event["data"]["decision"] == "approve"
    assert event["data"]["source"] == "api"


@pytest.mark.asyncio
async def test_api_respond_no_event_on_404():
    """T7a: approval.responded event is NOT emitted when respond fails (404)."""
    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    result = {}

    def do_request():
        client = _make_test_client()
        r = client.post("/api/approvals/nonexistent/respond", json={"decision": "approve"})
        result["status_code"] = r.status_code

    t = threading.Thread(target=do_request)
    t.start()
    t.join(timeout=5)

    assert result.get("status_code") == 404
    responded_events = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded_events) == 0


@pytest.mark.asyncio
async def test_api_respond_no_event_on_409():
    """T7b: approval.responded event is NOT emitted when respond fails (409 — already decided)."""
    approval_store.create_approval(
        "123:42", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"]
    )
    approval_store.respond("123:42", "approve")

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    result = {}

    def do_request():
        client = _make_test_client()
        r = client.post("/api/approvals/123:42/respond", json={"decision": "nosplit"})
        result["status_code"] = r.status_code

    t = threading.Thread(target=do_request)
    t.start()
    t.join(timeout=5)

    assert result.get("status_code") == 409
    responded_events = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded_events) == 0


@pytest.mark.asyncio
async def test_api_respond_no_event_on_400():
    """T7c: approval.responded event is NOT emitted when respond fails (400 — bad decision)."""
    approval_store.create_approval(
        "123:42", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"]
    )

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    result = {}

    def do_request():
        client = _make_test_client()
        r = client.post("/api/approvals/123:42/respond", json={"decision": "invalid_choice"})
        result["status_code"] = r.status_code

    t = threading.Thread(target=do_request)
    t.start()
    t.join(timeout=5)

    assert result.get("status_code") == 400
    responded_events = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded_events) == 0


# ── Group C: Timeout Auto-Decision ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_strategy_timeout_defaults_to_approve():
    """T8: Strategy approval timeout → decision defaults to 'approve'."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    approval_store.create_approval(
        "123:1", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"]
    )

    # Simulate the timeout path from handlers.py
    decision = await approval_store.wait_for_decision("123:1", timeout=0.05)
    assert decision is None  # timed out

    if decision is None:
        decision = "approve"
        await _emit_event(EventType.APPROVAL_RESPONDED, {
            "approval_id": "123:1",
            "decision": "approve",
            "source": "timeout",
            "timeout": True,
        })

    assert decision == "approve"

    responded = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded) == 1
    assert responded[0]["data"]["decision"] == "approve"
    assert responded[0]["data"]["source"] == "timeout"
    assert responded[0]["data"]["timeout"] is True


@pytest.mark.asyncio
async def test_supreme_court_timeout_defaults_to_ruling():
    """T9: Supreme Court approval timeout → decision defaults to ruling.lower()."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    ruling = "UPHOLD"
    approval_store.create_approval(
        "court:123:5", ApprovalType.SUPREME_COURT, ["accept", "uphold", "overturn"],
        context={"ruling": ruling, "issue_num": 5},
    )

    decision = await approval_store.wait_for_decision("court:123:5", timeout=0.05)
    assert decision is None  # timed out

    if decision is None or decision == "accept":
        decision = ruling.lower()
        await _emit_event(EventType.APPROVAL_RESPONDED, {
            "approval_id": "court:123:5",
            "decision": decision,
            "source": "timeout",
            "timeout": True,
        })

    assert decision == "uphold"

    responded = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded) == 1
    assert responded[0]["data"]["decision"] == "uphold"
    assert responded[0]["data"]["timeout"] is True


@pytest.mark.asyncio
async def test_timed_out_approval_has_expired_status():
    """T10: Timed-out approval has status=EXPIRED in store."""
    approval_store.create_approval("123:1", ApprovalType.STRATEGY, ["approve"])
    await approval_store.wait_for_decision("123:1", timeout=0.05)
    approval = approval_store.get_approval("123:1")
    assert approval is not None
    assert approval.status == ApprovalStatus.EXPIRED


@pytest.mark.asyncio
async def test_timeout_emits_responded_with_timeout_flag():
    """T11: approval.responded event is emitted on timeout with decision and timeout=True flag."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    approval_store.create_approval("123:1", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"])
    decision = await approval_store.wait_for_decision("123:1", timeout=0.05)
    assert decision is None

    await _emit_event(EventType.APPROVAL_RESPONDED, {
        "approval_id": "123:1",
        "decision": "approve",
        "source": "timeout",
        "timeout": True,
    })

    responded = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded) == 1
    data = responded[0]["data"]
    assert data["approval_id"] == "123:1"
    assert data["decision"] == "approve"
    assert data["source"] == "timeout"
    assert data["timeout"] is True


# ── Group D: Integration (WS + Timeout combined) ─────────────────────────────


@pytest.mark.asyncio
async def test_ws_receives_required_then_responded_after_timeout():
    """T12: WS client receives approval.required, then approval.responded after timeout."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    # Emit APPROVAL_REQUIRED
    approval_store.create_approval("123:1", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"])
    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "123:1",
        "type": "strategy",
        "decision_options": ["approve", "nosplit", "cancel"],
        "context": {},
    })

    # Wait with short timeout
    decision = await approval_store.wait_for_decision("123:1", timeout=0.05)
    assert decision is None

    # Emit APPROVAL_RESPONDED (timeout path)
    await _emit_event(EventType.APPROVAL_RESPONDED, {
        "approval_id": "123:1",
        "decision": "approve",
        "source": "timeout",
        "timeout": True,
    })

    event_types = [e["event_type"] for e in ws.received]
    assert "approval.required" in event_types
    assert "approval.responded" in event_types
    # Order: required before responded
    assert event_types.index("approval.required") < event_types.index("approval.responded")


@pytest.mark.asyncio
async def test_ws_receives_required_then_responded_after_user_decision():
    """T13: WS client receives approval.required, then approval.responded after user decision."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    approval_store.create_approval("123:1", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"])

    # Emit APPROVAL_REQUIRED
    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "123:1",
        "type": "strategy",
        "decision_options": ["approve", "nosplit", "cancel"],
        "context": {},
    })

    # Simulate user responding before timeout
    async def respond_soon():
        await asyncio.sleep(0.02)
        applied = approval_store.respond("123:1", "nosplit")
        if applied:
            await _emit_event(EventType.APPROVAL_RESPONDED, {
                "approval_id": "123:1",
                "decision": "nosplit",
                "source": "api",
            })

    await asyncio.gather(
        respond_soon(),
        approval_store.wait_for_decision("123:1", timeout=2.0),
    )

    event_types = [e["event_type"] for e in ws.received]
    assert "approval.required" in event_types
    assert "approval.responded" in event_types

    responded_events = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert responded_events[0]["data"]["decision"] == "nosplit"
    assert responded_events[0]["data"]["source"] == "api"


@pytest.mark.asyncio
async def test_rapid_respond_before_timeout_no_double_event():
    """T14: Rapid respond before timeout cancels the timeout path — no double event."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    ws = FakeWS()
    await bus.register(ws)

    approval_store.create_approval("123:1", ApprovalType.STRATEGY, ["approve", "nosplit", "cancel"])

    await _emit_event(EventType.APPROVAL_REQUIRED, {
        "approval_id": "123:1",
        "type": "strategy",
        "decision_options": ["approve", "nosplit", "cancel"],
        "context": {},
    })

    # Respond immediately (before wait_for_decision timeout)
    applied = approval_store.respond("123:1", "approve")
    assert applied is True
    await _emit_event(EventType.APPROVAL_RESPONDED, {
        "approval_id": "123:1",
        "decision": "approve",
        "source": "api",
    })

    # Now wait_for_decision should return immediately (already decided)
    # and NOT emit another event (decision is not None → timeout path skipped)
    decision = await approval_store.wait_for_decision("123:1", timeout=2.0)
    assert decision == "approve"  # resolved, not timed out

    if decision is None:  # This branch should NOT be taken
        await _emit_event(EventType.APPROVAL_RESPONDED, {
            "approval_id": "123:1",
            "decision": "approve",
            "source": "timeout",
            "timeout": True,
        })

    # Only one approval.responded event (from API, not from timeout)
    responded_events = [e for e in ws.received if e["event_type"] == "approval.responded"]
    assert len(responded_events) == 1
    assert responded_events[0]["data"]["source"] == "api"
