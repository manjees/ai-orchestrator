"""Tests for POST /api/commands/* endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

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


_TEST_API_KEY = "test-commands-key"

_PROJECTS = {
    "my-app": {"path": "/tmp/my-app"},
    "other-service": {"path": "/tmp/other-service"},
}


def _create_app(
    projects: dict | None = None,
    **settings_overrides,
) -> TestClient:
    settings_overrides.setdefault("dashboard_api_key", _TEST_API_KEY)
    settings = _make_settings(**settings_overrides)
    app = create_api_app(settings, projects=projects if projects is not None else _PROJECTS)
    return TestClient(app, headers={"Authorization": f"Bearer {_TEST_API_KEY}"})


def _auth_header(key: str = _TEST_API_KEY) -> dict:
    return {"Authorization": f"Bearer {key}"}


# ── C1: POST /api/commands/solve ─────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_returns_command_id_and_accepted(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my-app", "issues": [42]},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert "command_id" in data
    assert len(data["command_id"]) == 36  # UUID format


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_missing_issues_returns_422(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my-app"},  # missing "issues"
    )
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_empty_issues_returns_422(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my-app", "issues": []},
    )
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_unknown_project_returns_404(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "nonexistent", "issues": [1]},
    )
    assert resp.status_code == 404


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_with_mode_option(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my-app", "issues": [1, 2], "mode": "express"},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_with_parallel_option(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my-app", "issues": [1, 2], "parallel": True},
    )
    assert resp.status_code == 202


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_invalid_mode_returns_400(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my-app", "issues": [1], "mode": "turbo"},
    )
    assert resp.status_code == 400


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_solve_prefix_project_resolution(mock_bg):
    """Prefix match should work: 'my' resolves to 'my-app'."""
    client = _create_app(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.post(
        "/api/commands/solve",
        json={"project": "my", "issues": [1]},
    )
    assert resp.status_code == 202


# ── C2: POST /api/commands/retry ─────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_retry", new_callable=AsyncMock)
@patch("orchestrator.api.command_routes.load_checkpoint")
def test_retry_returns_command_id(mock_cp, mock_bg):
    mock_cp.return_value = {
        "ctx": {"project_name": "my-app", "issue_num": 42, "mode": "standard", "steps": []},
        "pipeline_mode": "standard",
        "failed_step_index": 2,
    }
    client = _create_app()
    resp = client.post(
        "/api/commands/retry",
        json={"project": "my-app", "issue_num": 42},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
    assert "command_id" in resp.json()


@patch("orchestrator.api.command_routes._bg_retry", new_callable=AsyncMock)
def test_retry_missing_project_returns_422(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/retry",
        json={"issue_num": 42},  # missing "project"
    )
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_retry", new_callable=AsyncMock)
@patch("orchestrator.api.command_routes.load_checkpoint", return_value=None)
def test_retry_no_checkpoint_returns_404(mock_cp, mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/retry",
        json={"project": "my-app", "issue_num": 99},
    )
    assert resp.status_code == 404


@patch("orchestrator.api.command_routes._bg_retry", new_callable=AsyncMock)
def test_retry_unknown_project_returns_404(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/retry",
        json={"project": "ghost", "issue_num": 1},
    )
    assert resp.status_code == 404


# ── C3: POST /api/commands/cancel ────────────────────────────────────────────


def test_cancel_unknown_pipeline_returns_404():
    from orchestrator.api import registry
    registry.clear()
    client = _create_app()
    resp = client.post(
        "/api/commands/cancel",
        json={"pipeline_id": "my-app_99"},
    )
    assert resp.status_code == 404
    registry.clear()


def test_cancel_returns_accepted():
    from orchestrator.api import registry
    from orchestrator.api.command_routes import _api_cancel_events

    registry.clear()
    _api_cancel_events.clear()

    import asyncio
    cancel_event = asyncio.Event()
    pid = "my-app_42"

    # Simulate a running pipeline: register in both registry and cancel dict
    registry._active[pid] = {
        "project_name": "my-app", "issue_num": 42,
        "mode": "standard", "steps": [],
    }
    _api_cancel_events[pid] = cancel_event

    client = _create_app()
    resp = client.post("/api/commands/cancel", json={"pipeline_id": pid})
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
    assert cancel_event.is_set()

    registry.clear()
    _api_cancel_events.clear()


def test_cancel_missing_pipeline_id_returns_422():
    client = _create_app()
    resp = client.post("/api/commands/cancel", json={})
    assert resp.status_code == 422


# ── C4: POST /api/commands/init ──────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_init", new_callable=AsyncMock)
def test_init_returns_command_id(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/init",
        json={"name": "new-project", "description": "A brand new project"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert "command_id" in data


@patch("orchestrator.api.command_routes._bg_init", new_callable=AsyncMock)
def test_init_missing_fields_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/init", json={"name": "new-project"})
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_init", new_callable=AsyncMock)
def test_init_invalid_project_name_returns_422(mock_bg):
    """Pydantic pattern validation rejects invalid names."""
    client = _create_app()
    for bad_name in ["My App", "-bad", "bad!", "UPPER"]:
        resp = client.post(
            "/api/commands/init",
            json={"name": bad_name, "description": "test"},
        )
        assert resp.status_code == 422, f"Expected 422 for name={bad_name!r}"


@patch("orchestrator.api.command_routes._bg_init", new_callable=AsyncMock)
def test_init_duplicate_project_returns_409(mock_bg):
    """Should return 409 if project name already exists in projects dict."""
    client = _create_app(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.post(
        "/api/commands/init",
        json={"name": "my-app", "description": "duplicate"},
    )
    assert resp.status_code == 409


@patch("orchestrator.api.command_routes._bg_init", new_callable=AsyncMock)
def test_init_missing_github_user_returns_400(mock_bg):
    """Should return 400 if GITHUB_USER is not configured."""
    client = _create_app(github_user="")
    resp = client.post(
        "/api/commands/init",
        json={"name": "new-app", "description": "test"},
    )
    assert resp.status_code == 400


# ── C5: POST /api/commands/plan ──────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_plan", new_callable=AsyncMock)
def test_plan_returns_command_id(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/plan", json={"project": "my-app"})
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
    assert "command_id" in resp.json()


@patch("orchestrator.api.command_routes._bg_plan", new_callable=AsyncMock)
def test_plan_unknown_project_returns_404(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/plan", json={"project": "ghost"})
    assert resp.status_code == 404


@patch("orchestrator.api.command_routes._bg_plan", new_callable=AsyncMock)
def test_plan_missing_project_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/plan", json={})
    assert resp.status_code == 422


# ── C6: POST /api/commands/discuss ───────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_discuss", new_callable=AsyncMock)
def test_discuss_returns_command_id(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/discuss",
        json={"project": "my-app", "question": "How should we handle auth?"},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


@patch("orchestrator.api.command_routes._bg_discuss", new_callable=AsyncMock)
def test_discuss_missing_question_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/discuss", json={"project": "my-app"})
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_discuss", new_callable=AsyncMock)
def test_discuss_unknown_project_returns_404(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/discuss",
        json={"project": "ghost", "question": "What?"},
    )
    assert resp.status_code == 404


# ── C7: POST /api/commands/design ────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_design", new_callable=AsyncMock)
def test_design_returns_command_id(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/design",
        json={"project": "my-app", "figma_url": "https://figma.com/file/abc/test"},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


@patch("orchestrator.api.command_routes._bg_design", new_callable=AsyncMock)
def test_design_missing_figma_url_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/design", json={"project": "my-app"})
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_design", new_callable=AsyncMock)
def test_design_missing_figma_token_returns_400(mock_bg):
    client = _create_app(figma_access_token="")
    resp = client.post(
        "/api/commands/design",
        json={"project": "my-app", "figma_url": "https://figma.com/file/abc"},
    )
    assert resp.status_code == 400


@patch("orchestrator.api.command_routes._bg_design", new_callable=AsyncMock)
def test_design_unknown_project_returns_404(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/design",
        json={"project": "ghost", "figma_url": "https://figma.com/file/abc"},
    )
    assert resp.status_code == 404


# ── C8: POST /api/commands/rebase ────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_rebase", new_callable=AsyncMock)
def test_rebase_returns_command_id(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/rebase",
        json={"project": "my-app", "pr_number": 7},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


@patch("orchestrator.api.command_routes._bg_rebase", new_callable=AsyncMock)
def test_rebase_missing_pr_number_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/rebase", json={"project": "my-app"})
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_rebase", new_callable=AsyncMock)
def test_rebase_unknown_project_returns_404(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/rebase",
        json={"project": "ghost", "pr_number": 1},
    )
    assert resp.status_code == 404


# ── C9: POST /api/commands/shell ─────────────────────────────────────────────


@patch("orchestrator.api.command_routes._bg_shell", new_callable=AsyncMock)
def test_shell_returns_command_id(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/shell", json={"command": "ls -la"})
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "accepted"
    assert "command_id" in data


@patch("orchestrator.api.command_routes._bg_shell", new_callable=AsyncMock)
def test_shell_dangerous_command_returns_warning(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/shell", json={"command": "rm -rf /"})
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "warning"
    assert "warning" in data
    assert data["warning"]  # non-empty


@patch("orchestrator.api.command_routes._bg_shell", new_callable=AsyncMock)
def test_shell_with_timeout_option(mock_bg):
    client = _create_app()
    resp = client.post(
        "/api/commands/shell",
        json={"command": "sleep 5", "timeout": 10},
    )
    assert resp.status_code == 202


@patch("orchestrator.api.command_routes._bg_shell", new_callable=AsyncMock)
def test_shell_empty_command_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/shell", json={"command": ""})
    assert resp.status_code == 422


@patch("orchestrator.api.command_routes._bg_shell", new_callable=AsyncMock)
def test_shell_missing_command_returns_422(mock_bg):
    client = _create_app()
    resp = client.post("/api/commands/shell", json={})
    assert resp.status_code == 422


# ── C10: Common behaviors ─────────────────────────────────────────────────────


def test_all_endpoints_require_auth():
    """All command endpoints return 401 when no API key is set and no auth provided."""
    settings = _make_settings(dashboard_api_key="secret-key")
    from orchestrator.api.app import create_api_app
    app = create_api_app(settings, projects=_PROJECTS)
    client = TestClient(app)  # no auth header

    endpoints = [
        ("/api/commands/solve", {"project": "my-app", "issues": [1]}),
        ("/api/commands/retry", {"project": "my-app", "issue_num": 1}),
        ("/api/commands/cancel", {"pipeline_id": "my-app_1"}),
        ("/api/commands/init", {"name": "new-app", "description": "test"}),
        ("/api/commands/plan", {"project": "my-app"}),
        ("/api/commands/discuss", {"project": "my-app", "question": "Why?"}),
        ("/api/commands/design", {"project": "my-app", "figma_url": "https://figma.com/x"}),
        ("/api/commands/rebase", {"project": "my-app", "pr_number": 1}),
        ("/api/commands/shell", {"command": "ls"}),
    ]
    for path, body in endpoints:
        resp = client.post(path, json=body)
        assert resp.status_code == 401, f"Expected 401 for {path}, got {resp.status_code}"


@patch("orchestrator.api.command_routes._bg_solve", new_callable=AsyncMock)
def test_command_id_is_unique_uuid(mock_bg):
    client = _create_app()
    ids = set()
    for _ in range(5):
        resp = client.post(
            "/api/commands/solve",
            json={"project": "my-app", "issues": [1]},
        )
        assert resp.status_code == 202
        ids.add(resp.json()["command_id"])
    assert len(ids) == 5  # All unique
