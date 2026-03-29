"""Tests for GET /api/checkpoints endpoint."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from orchestrator.api.app import create_api_app
from orchestrator.config import Settings


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key="test-key",
        cors_origins="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture()
def client():
    app = create_api_app(_make_settings())
    return TestClient(app)


@pytest.fixture()
def auth_header():
    return {"Authorization": "Bearer test-key"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _checkpoint(
    file="myapp_42.json",
    project_name="myapp",
    issue_num=42,
    pipeline_mode="standard",
    failed_step_name="Sonnet Implement",
):
    return dict(
        file=file,
        project_name=project_name,
        issue_num=issue_num,
        pipeline_mode=pipeline_mode,
        failed_step_name=failed_step_name,
    )


# ── T1: Normal 200 response ─────────────────────────────────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_list_checkpoints_200(mock_list, client, auth_header):
    """Returns 200 with list of CheckpointSummary objects."""
    mock_list.return_value = [_checkpoint()]
    resp = client.get("/api/checkpoints", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["file"] == "myapp_42.json"
    assert data[0]["failed_step_name"] == "Sonnet Implement"


# ── T2: Empty list ──────────────────────────────────────────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_list_checkpoints_empty(mock_list, client, auth_header):
    """No checkpoints → 200, empty list."""
    mock_list.return_value = []
    resp = client.get("/api/checkpoints", headers=auth_header)
    assert resp.status_code == 200
    assert resp.json() == []


# ── T3: Multiple checkpoints ────────────────────────────────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_list_checkpoints_multiple(mock_list, client, auth_header):
    """Multiple checkpoints → all returned."""
    mock_list.return_value = [
        _checkpoint(file="a_1.json", project_name="a", issue_num=1),
        _checkpoint(file="b_2.json", project_name="b", issue_num=2),
        _checkpoint(file="c_3.json", project_name="c", issue_num=3),
    ]
    resp = client.get("/api/checkpoints", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    names = {item["project_name"] for item in data}
    assert names == {"a", "b", "c"}


# ── T4: Response schema validation ──────────────────────────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_list_checkpoints_response_schema(mock_list, client, auth_header):
    """Each item has all 5 required fields."""
    mock_list.return_value = [_checkpoint()]
    resp = client.get("/api/checkpoints", headers=auth_header)
    data = resp.json()
    expected_keys = {"file", "project_name", "issue_num", "pipeline_mode", "failed_step_name"}
    assert set(data[0].keys()) == expected_keys


# ── T5: Secret masking in failed_step_name ───────────────────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_list_checkpoints_secret_masking(mock_list, client, auth_header):
    """Secret in failed_step_name → replaced with [MASKED]."""
    mock_list.return_value = [
        _checkpoint(failed_step_name="step with key sk-ant-abc123456789"),
    ]
    resp = client.get("/api/checkpoints", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "sk-ant-" not in data[0]["failed_step_name"]
    assert "[MASKED]" in data[0]["failed_step_name"]


# ── T6: Secret masking in file field ────────────────────────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_list_checkpoints_secret_masking_file_field(mock_list, client, auth_header):
    """Secret in file field → masked."""
    mock_list.return_value = [
        _checkpoint(file="ghp_abcdefghij1234567890abcdefghij123456.json"),
    ]
    resp = client.get("/api/checkpoints", headers=auth_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "ghp_" not in data[0]["file"]
    assert "[MASKED]" in data[0]["file"]


# ── T7: Auth required ───────────────────────────────────────────────────────


def test_list_checkpoints_requires_auth(client):
    """No auth header → 401."""
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 401
