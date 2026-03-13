"""Tests for orchestrator.dashboard_client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from orchestrator.config import Settings
from orchestrator.dashboard_client import AGENT_REGISTRY, DashboardClient

BASE_URL = "http://localhost:8000"


# ── Group A: Config defaults ────────────────────────────────────────────────


def test_dashboard_disabled_by_default():
    s = Settings(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
    )
    assert s.dashboard_api_url == ""
    assert s.dashboard_api_timeout == 5
    assert s.dashboard_api_key == ""


# ── Group B: Client creation & lifecycle ────────────────────────────────────


def test_client_creation_sets_base_url_and_timeout():
    client = DashboardClient(base_url=BASE_URL, timeout=10)
    assert client._base_url == BASE_URL
    assert client._client.timeout.connect == 10.0
    assert client._client.timeout.read == 10.0


def test_client_creation_with_api_key_sets_auth_header():
    client = DashboardClient(base_url=BASE_URL, api_key="secret-token")
    assert client._client.headers["authorization"] == "Bearer secret-token"


def test_client_creation_without_api_key_has_no_auth_header():
    client = DashboardClient(base_url=BASE_URL)
    assert "authorization" not in client._client.headers


async def test_close_calls_aclose():
    client = DashboardClient(base_url=BASE_URL)
    client._client.aclose = AsyncMock()
    await client.close()
    client._client.aclose.assert_awaited_once()


# ── Group C: register_agents() — POST /api/v1/agents ───────────────────────


@respx.mock
async def test_register_agents_posts_all_seven():
    route = respx.post(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(200, json={"id": "test"})
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.register_agents()
    assert route.call_count == len(AGENT_REGISTRY)
    await client.close()


@respx.mock
async def test_register_agents_server_error_does_not_raise():
    respx.post(f"{BASE_URL}/api/v1/agents").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.register_agents()
    await client.close()


@respx.mock
async def test_register_agents_connection_error_does_not_raise():
    respx.post(f"{BASE_URL}/api/v1/agents").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.register_agents()
    await client.close()


# ── Group D: update_agent_status() — PATCH /api/v1/agents/{id}/status ──────


@respx.mock
async def test_update_agent_status_sends_patch():
    route = respx.patch(f"{BASE_URL}/api/v1/agents/sonnet-worker/status").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.update_agent_status("sonnet-worker", "RUNNING")
    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body == {"status": "RUNNING"}
    await client.close()


@respx.mock
async def test_update_agent_status_error_does_not_raise():
    respx.patch(f"{BASE_URL}/api/v1/agents/sonnet-worker/status").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.update_agent_status("sonnet-worker", "RUNNING")
    await client.close()


async def test_update_agent_status_rejects_empty_agent_id():
    client = DashboardClient(base_url=BASE_URL)
    with pytest.raises(ValueError, match="agent_id"):
        await client.update_agent_status("", "RUNNING")
    await client.close()


async def test_update_agent_status_rejects_empty_status():
    client = DashboardClient(base_url=BASE_URL)
    with pytest.raises(ValueError, match="status"):
        await client.update_agent_status("sonnet-worker", "")
    await client.close()


# ── Group E: send_event() — POST /api/v1/events ────────────────────────────


@respx.mock
async def test_send_event_posts_payload():
    route = respx.post(f"{BASE_URL}/api/v1/events").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.send_event("PIPELINE_STARTED", {"issue": 42})
    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body["event_type"] == "PIPELINE_STARTED"
    assert body["data"] == {"issue": 42}
    await client.close()


@respx.mock
async def test_send_event_error_does_not_raise():
    respx.post(f"{BASE_URL}/api/v1/events").mock(
        side_effect=httpx.ReadTimeout("Timeout")
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.send_event("PIPELINE_STARTED")
    await client.close()


async def test_send_event_rejects_empty_event_type():
    client = DashboardClient(base_url=BASE_URL)
    with pytest.raises(ValueError, match="event_type"):
        await client.send_event("")
    await client.close()


# ── Group F: create_pipeline_run() / update_pipeline_run() ─────────────────


@respx.mock
async def test_create_pipeline_run_posts_and_returns_id():
    respx.post(f"{BASE_URL}/api/v1/pipeline-runs").mock(
        return_value=httpx.Response(200, json={"id": "run-abc-123"})
    )
    client = DashboardClient(base_url=BASE_URL)
    run_id = await client.create_pipeline_run({"issue": 1})
    assert run_id == "run-abc-123"
    await client.close()


@respx.mock
async def test_create_pipeline_run_error_returns_none():
    respx.post(f"{BASE_URL}/api/v1/pipeline-runs").mock(
        side_effect=httpx.ConnectError("refused")
    )
    client = DashboardClient(base_url=BASE_URL)
    run_id = await client.create_pipeline_run({"issue": 1})
    assert run_id is None
    await client.close()


@respx.mock
async def test_update_pipeline_run_sends_patch():
    route = respx.patch(f"{BASE_URL}/api/v1/pipeline-runs/run-abc-123").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = DashboardClient(base_url=BASE_URL)
    await client.update_pipeline_run("run-abc-123", {"step": "design", "status": "completed"})
    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body["step"] == "design"
    assert body["status"] == "completed"
    await client.close()


async def test_update_pipeline_run_rejects_empty_run_id():
    client = DashboardClient(base_url=BASE_URL)
    with pytest.raises(ValueError, match="run_id"):
        await client.update_pipeline_run("", {"step": "done"})
    await client.close()


# ── Group G: Fire-and-forget guarantee ──────────────────────────────────────


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("register_agents", ()),
        ("update_agent_status", ("orchestrator", "RUNNING")),
        ("send_event", ("ERROR", {"msg": "boom"})),
        ("create_pipeline_run", ({"issue": 1},)),
        ("update_pipeline_run", ("run-1", {"step": "done"})),
    ],
)
@respx.mock
async def test_all_methods_swallow_exceptions(method_name, args):
    """Every public method catches network/HTTP exceptions without propagating."""
    respx.route().mock(side_effect=RuntimeError("unexpected"))
    client = DashboardClient(base_url=BASE_URL)
    method = getattr(client, method_name)
    await method(*args)
    await client.close()


# ── Group H: Disabled client (NullObject pattern) ──────────────────────────


def test_no_ops_when_disabled():
    s = Settings(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
    )
    assert s.dashboard_api_url == ""
    assert not s.dashboard_api_url


# ── Group I: Auth header sent on requests ───────────────────────────────────


@respx.mock
async def test_api_key_sent_as_bearer_token_on_requests():
    route = respx.post(f"{BASE_URL}/api/v1/events").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = DashboardClient(base_url=BASE_URL, api_key="my-secret")
    await client.send_event("TEST_EVENT")
    auth_header = route.calls[0].request.headers["authorization"]
    assert auth_header == "Bearer my-secret"
    await client.close()
