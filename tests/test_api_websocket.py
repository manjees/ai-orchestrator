"""Tests for WebSocket /ws/events endpoint and EventBus."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchestrator.api.app import create_api_app
from orchestrator.api.events import EventBus, EventType, WSEvent, get_event_bus, reset_event_bus
from orchestrator.config import Settings


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key="",
        cors_origins="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


# ── Group A: Authentication ─────────────────────────────────────────────────


def test_ws_rejects_connection_without_token():
    """WS connect without token when API key is configured -> close 1008."""
    app = create_api_app(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/api/ws/events"):
            pass


def test_ws_rejects_connection_with_invalid_token():
    """WS connect with wrong token -> close 1008."""
    app = create_api_app(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/api/ws/events?token=wrong-key"):
            pass


def test_ws_accepts_connection_with_valid_token():
    """WS connect with correct token -> accepted."""
    app = create_api_app(_make_settings(dashboard_api_key="secret123"))
    client = TestClient(app)
    with client.websocket_connect("/api/ws/events?token=secret123") as ws:
        # Connection accepted — no exception raised
        assert ws is not None


def test_ws_accepts_connection_when_no_api_key_configured():
    """WS connect with no API key configured -> accepted (open mode)."""
    app = create_api_app(_make_settings(dashboard_api_key=""))
    client = TestClient(app)
    with client.websocket_connect("/api/ws/events") as ws:
        assert ws is not None


# ── Group B: EventBus Core ──────────────────────────────────────────────────


def test_eventbus_is_singleton():
    """get_event_bus() returns the same instance on repeated calls."""
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2


def test_eventbus_register_and_unregister_client():
    """Register adds client; unregister removes it."""
    bus = get_event_bus()

    class FakeWS:
        pass

    ws = FakeWS()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(bus.register(ws))
        assert bus.client_count == 1
        loop.run_until_complete(bus.unregister(ws))
        assert bus.client_count == 0
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_eventbus_emit_sends_to_all_clients():
    """Emit broadcasts to all registered clients."""
    bus = get_event_bus()
    received = []

    class FakeWS:
        async def send_text(self, data):
            received.append(data)

    ws1, ws2 = FakeWS(), FakeWS()
    await bus.register(ws1)
    await bus.register(ws2)
    await bus.emit("test.event", {"key": "value"})

    assert len(received) == 2
    for msg in received:
        parsed = json.loads(msg)
        assert parsed["event_type"] == "test.event"
        assert parsed["data"]["key"] == "value"
        assert "timestamp" in parsed


@pytest.mark.asyncio
async def test_eventbus_emit_removes_disconnected_client():
    """Dead client is removed on emit; live client still receives."""
    bus = get_event_bus()
    live_received = []

    class LiveWS:
        async def send_text(self, data):
            live_received.append(data)

    class DeadWS:
        async def send_text(self, data):
            raise ConnectionError("gone")

    live, dead = LiveWS(), DeadWS()
    await bus.register(live)
    await bus.register(dead)
    assert bus.client_count == 2

    await bus.emit("test.event")

    assert len(live_received) == 1
    assert bus.client_count == 1


@pytest.mark.asyncio
async def test_eventbus_emit_with_no_clients_does_not_raise():
    """Emit with no clients is a no-op."""
    bus = get_event_bus()
    await bus.emit("test.event", {"some": "data"})
    # No exception raised


# ── Group C: WebSocket Endpoint Integration ─────────────────────────────────


def test_ws_receives_emitted_event():
    """Connected client receives events emitted via EventBus."""
    app = create_api_app(_make_settings(dashboard_api_key=""))
    client = TestClient(app)

    with client.websocket_connect("/api/ws/events") as ws:
        bus = get_event_bus()
        # Emit from a background thread since the WS receive blocks
        import threading

        def emit_event():
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            loop.run_until_complete(bus.emit(EventType.PIPELINE_STARTED, {"pipeline_id": "p1"}))
            loop.close()

        t = threading.Thread(target=emit_event)
        t.start()
        t.join(timeout=5)

        data = ws.receive_json(mode="text")
        assert data["event_type"] == "pipeline.started"
        assert data["data"]["pipeline_id"] == "p1"
        assert "timestamp" in data


def test_ws_event_has_correct_schema():
    """Emitted event matches the WSEvent schema."""
    app = create_api_app(_make_settings(dashboard_api_key=""))
    client = TestClient(app)

    with client.websocket_connect("/api/ws/events") as ws:
        bus = get_event_bus()
        import threading

        def emit_event():
            import asyncio as _asyncio
            loop = _asyncio.new_event_loop()
            loop.run_until_complete(bus.emit(
                EventType.STEP_COMPLETED,
                {"step": "Opus Design", "status": "passed", "elapsed_sec": 12.5},
            ))
            loop.close()

        t = threading.Thread(target=emit_event)
        t.start()
        t.join(timeout=5)

        data = ws.receive_json(mode="text")
        assert data["event_type"] == "step.completed"
        assert data["data"]["step"] == "Opus Design"
        assert data["data"]["status"] == "passed"
        assert data["data"]["elapsed_sec"] == 12.5
        # Validate WSEvent can parse it
        event = WSEvent(**data)
        assert event.event_type == "step.completed"


def test_ws_multiple_clients_receive_broadcast():
    """Multiple connected clients all receive the same event."""
    app = create_api_app(_make_settings(dashboard_api_key=""))
    client1 = TestClient(app)
    client2 = TestClient(app)

    with client1.websocket_connect("/api/ws/events") as ws1:
        with client2.websocket_connect("/api/ws/events") as ws2:
            bus = get_event_bus()
            import threading

            def emit_event():
                import asyncio as _asyncio
                loop = _asyncio.new_event_loop()
                loop.run_until_complete(bus.emit("broadcast.test", {"msg": "hello"}))
                loop.close()

            t = threading.Thread(target=emit_event)
            t.start()
            t.join(timeout=5)

            d1 = ws1.receive_json(mode="text")
            d2 = ws2.receive_json(mode="text")
            assert d1["event_type"] == "broadcast.test"
            assert d2["event_type"] == "broadcast.test"
            assert d1["data"]["msg"] == "hello"
            assert d2["data"]["msg"] == "hello"


# ── Group D: Event Types ───────────────────────────────────────────────────


def test_all_event_types_defined():
    """All 8 event types are defined as constants."""
    assert EventType.PIPELINE_STARTED == "pipeline.started"
    assert EventType.STEP_STARTED == "step.started"
    assert EventType.STEP_COMPLETED == "step.completed"
    assert EventType.STEP_FAILED == "step.failed"
    assert EventType.PIPELINE_COMPLETED == "pipeline.completed"
    assert EventType.PIPELINE_FAILED == "pipeline.failed"
    assert EventType.APPROVAL_REQUIRED == "approval.required"
    assert EventType.SYSTEM_STATUS == "system.status"


def test_ws_event_model_fields():
    """WSEvent has event_type, data, and timestamp fields."""
    event = WSEvent(event_type="test", data={"k": "v"}, timestamp="2026-01-01T00:00:00+00:00")
    assert event.event_type == "test"
    assert event.data == {"k": "v"}
    assert event.timestamp == "2026-01-01T00:00:00+00:00"


# ── Group E: Pipeline Integration ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_emit_event_helper():
    """_emit_event helper sends events via EventBus without raising."""
    from orchestrator.pipeline import _emit_event

    bus = get_event_bus()
    received = []

    class FakeWS:
        async def send_text(self, data):
            received.append(data)

    ws = FakeWS()
    await bus.register(ws)

    await _emit_event("pipeline.started", {"pipeline_id": "test_1", "project_name": "myproj"})
    assert len(received) == 1
    parsed = json.loads(received[0])
    assert parsed["event_type"] == "pipeline.started"
    assert parsed["data"]["pipeline_id"] == "test_1"


@pytest.mark.asyncio
async def test_emit_event_does_not_raise_on_failure():
    """_emit_event silently swallows exceptions."""
    from orchestrator.pipeline import _emit_event

    # Even with a broken bus, no exception should propagate
    reset_event_bus()
    bus = get_event_bus()

    class BadWS:
        async def send_text(self, data):
            raise RuntimeError("send failed")

    await bus.register(BadWS())
    # Should not raise
    await _emit_event("pipeline.failed", {"detail": "boom"})


# ── Group F: System Status Loop ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_status_loop_emits_events(monkeypatch):
    """system.status events are emitted periodically."""
    from orchestrator.system_monitor import SystemStatus

    fake_status = SystemStatus(
        ram_total_gb=16.0, ram_used_gb=8.0, ram_percent=50.0,
        cpu_percent=25.0, thermal_pressure="nominal",
        disk_total_gb=500.0, disk_used_gb=250.0, disk_percent=50.0,
    )

    async def mock_get_status():
        return fake_status

    monkeypatch.setattr("orchestrator.api.events.get_system_status", mock_get_status)
    monkeypatch.setattr("orchestrator.api.events.registry_list_all", lambda: {})

    bus = get_event_bus()
    received = []

    class FakeWS:
        async def send_text(self, data):
            received.append(data)

    await bus.register(FakeWS())
    await bus.start_status_loop(interval=0.1)

    # Wait enough for at least 1 emission
    await asyncio.sleep(0.35)
    await bus.stop_status_loop()

    assert len(received) >= 1
    parsed = json.loads(received[0])
    assert parsed["event_type"] == "system.status"
    assert parsed["data"]["ram_percent"] == 50.0
    assert parsed["data"]["cpu_percent"] == 25.0
