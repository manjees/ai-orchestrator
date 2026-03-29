"""Tests for GET endpoints: status, projects, pipelines, checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

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


_TEST_API_KEY = "test-key-for-endpoints"


def _create_app(
    projects: dict | None = None,
    pipelines: dict | None = None,
    **settings_overrides,
) -> TestClient:
    settings_overrides.setdefault("dashboard_api_key", _TEST_API_KEY)
    settings = _make_settings(**settings_overrides)
    app = create_api_app(settings, projects=projects or {}, pipelines=pipelines or {})
    return TestClient(app, headers={"Authorization": f"Bearer {settings.dashboard_api_key}"})


# ── Helpers: mock pipeline context ───────────────────────────────────────────


@dataclass
class _MockStep:
    name: str = "Sonnet Implement"
    status: str = "passed"
    detail: str = "ok"
    elapsed_sec: float = 1.5


@dataclass
class _MockPipelineCtx:
    id: str = "pipe-1"
    project_name: str = "my-app"
    issue_num: int = 42
    branch_name: str = "solve/issue-42"
    mode: str = "standard"
    issue_title: str = "Fix bug"
    issue_body: str = ""
    design_doc: str = ""
    git_diff: str = ""
    ci_check_log: str = ""
    review_report: str = ""
    ai_audit_result: str = ""
    ai_audit_passed: bool = False
    steps: list = field(default_factory=lambda: [_MockStep()])


# ── D1: GET /api/status — 200 with system metrics ───────────────────────────


@patch("orchestrator.api.routes.list_sessions", new_callable=AsyncMock)
@patch("orchestrator.api.routes.get_system_status", new_callable=AsyncMock)
def test_status_200(mock_status, mock_tmux):
    mock_status.return_value = type(
        "SystemStatus", (), dict(
            ram_total_gb=16.0, ram_used_gb=8.0, ram_percent=50.0,
            cpu_percent=25.0, thermal_pressure="nominal",
            disk_total_gb=500.0, disk_used_gb=200.0, disk_percent=40.0,
        ),
    )()
    mock_tmux.return_value = [
        type("TmuxSession", (), dict(name="main", windows=3, created="1711700000"))(),
    ]
    client = _create_app()
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ram_total_gb"] == 16.0
    assert data["cpu_percent"] == 25.0
    assert data["thermal_pressure"] == "nominal"
    assert data["disk_percent"] == 40.0
    assert len(data["tmux_sessions"]) == 1
    assert data["tmux_sessions"][0]["name"] == "main"
    assert data["tmux_sessions"][0]["windows"] == 3


# ── D2: GET /api/status — requires auth when api_key set ────────────────────


def test_status_requires_auth():
    settings = _make_settings(dashboard_api_key="secret")
    app = create_api_app(settings)
    client = TestClient(app)  # no auth header
    resp = client.get("/api/status")
    assert resp.status_code == 401


# ── D3: GET /api/projects — 200 with project list ───────────────────────────


def test_projects_200():
    client = _create_app(projects={"my-app": {"path": "/home/user/my-app"}})
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "my-app"
    assert data[0]["path"] == "/home/user/my-app"


# ── D4: GET /api/projects — empty list when no projects ─────────────────────


def test_projects_empty():
    client = _create_app(projects={})
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


# ── D5: GET /api/projects/{name} — 200 with project detail ──────────────────


@patch("orchestrator.api.routes.load_project_summary", return_value={"key": "value"})
def test_project_detail_200(mock_load):
    client = _create_app(projects={"my-app": {"path": "/home/user/my-app"}})
    resp = client.get("/api/projects/my-app")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "my-app"
    assert data["path"] == "/home/user/my-app"
    assert data["summary"] == {"key": "value"}


# ── D6: GET /api/projects/{name} — 404 when project not found ───────────────


def test_project_detail_404():
    client = _create_app(projects={})
    resp = client.get("/api/projects/nonexistent")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Project not found"


# ── D7: GET /api/projects/{name}/issues — 200 with issue list ───────────────


@patch("orchestrator.api.routes._fetch_github_issues", new_callable=AsyncMock)
def test_project_issues_200(mock_fetch):
    mock_fetch.return_value = [
        {"number": 1, "title": "Bug", "labels": ["bug"], "url": "https://github.com/x/1"},
    ]
    client = _create_app(projects={"my-app": {"path": "/home/user/my-app"}})
    resp = client.get("/api/projects/my-app/issues")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["number"] == 1
    assert data[0]["title"] == "Bug"


# ── D7b: gh label objects are normalised to name strings ────────────────────


@patch("orchestrator.api.routes._fetch_github_issues", new_callable=AsyncMock)
def test_project_issues_label_objects_normalised(mock_fetch):
    """gh CLI returns labels as dicts; ensure they are flattened to strings."""
    mock_fetch.return_value = [
        {
            "number": 2,
            "title": "Feature",
            "labels": [{"id": "LA_1", "name": "enhancement", "color": "a2eeef"}],
            "url": "https://github.com/x/2",
        },
    ]
    client = _create_app(projects={"my-app": {"path": "/home/user/my-app"}})
    resp = client.get("/api/projects/my-app/issues")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["labels"] == ["enhancement"]


# ── D8: GET /api/projects/{name}/issues — 404 for unknown project ───────────


def test_project_issues_404():
    client = _create_app(projects={})
    resp = client.get("/api/projects/nonexistent/issues")
    assert resp.status_code == 404


# ── D9: GET /api/projects/{name}/issues — empty list when gh fails ───────────


@patch("orchestrator.api.routes._fetch_github_issues", new_callable=AsyncMock)
def test_project_issues_gh_fails(mock_fetch):
    mock_fetch.return_value = []
    client = _create_app(projects={"my-app": {"path": "/home/user/my-app"}})
    resp = client.get("/api/projects/my-app/issues")
    assert resp.status_code == 200
    assert resp.json() == []


# ── D10: GET /api/pipelines — 200 with pipeline list ────────────────────────


@patch("orchestrator.api.routes.list_checkpoints", return_value=[])
def test_pipelines_200(mock_cp):
    from orchestrator.api import registry
    registry.clear()
    ctx = _MockPipelineCtx()
    registry._active["my-app_42"] = dict(
        project_name=ctx.project_name, issue_num=ctx.issue_num,
        branch_name=ctx.branch_name, mode=ctx.mode, issue_title=ctx.issue_title,
        steps=[dict(name=s.name, status=s.status, detail=s.detail, elapsed_sec=s.elapsed_sec) for s in ctx.steps],
    )
    client = _create_app()
    resp = client.get("/api/pipelines")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pipelines"]) == 1
    p = data["pipelines"][0]
    assert p["pipeline_id"] == "my-app_42"
    assert p["project_name"] == "my-app"
    assert p["issue_num"] == 42
    assert p["mode"] == "standard"
    assert len(p["steps"]) == 1
    registry.clear()


# ── D11: GET /api/pipelines — empty when no pipelines ───────────────────────


@patch("orchestrator.api.routes.list_checkpoints", return_value=[])
def test_pipelines_empty(mock_cp):
    from orchestrator.api import registry
    registry.clear()
    client = _create_app()
    resp = client.get("/api/pipelines")
    assert resp.status_code == 200
    assert resp.json() == {"pipelines": []}
    registry.clear()


# ── D12: GET /api/pipelines/{id} — 200 with pipeline detail ─────────────────


def test_pipeline_detail_200():
    from orchestrator.api import registry
    registry.clear()
    ctx = _MockPipelineCtx()
    registry._active["my-app_42"] = dict(
        project_name=ctx.project_name, issue_num=ctx.issue_num,
        branch_name=ctx.branch_name, mode=ctx.mode, issue_title=ctx.issue_title,
        issue_body=ctx.issue_body, design_doc=ctx.design_doc, git_diff=ctx.git_diff,
        review_report=ctx.review_report, ai_audit_result=ctx.ai_audit_result,
        ci_check_log=ctx.ci_check_log,
        steps=[dict(name=s.name, status=s.status, detail=s.detail, elapsed_sec=s.elapsed_sec) for s in ctx.steps],
    )
    client = _create_app()
    resp = client.get("/api/pipelines/my-app_42")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_id"] == "my-app_42"
    assert data["project_name"] == "my-app"
    assert data["branch_name"] == "solve/issue-42"
    assert data["issue_title"] == "Fix bug"
    assert len(data["steps"]) == 1
    registry.clear()


# ── D13: GET /api/pipelines/{id} — 404 when not found ───────────────────────


@patch("orchestrator.api.routes.load_checkpoint", return_value=None)
def test_pipeline_detail_404(mock_cp):
    from orchestrator.api import registry
    registry.clear()
    client = _create_app()
    resp = client.get("/api/pipelines/nonexistent_1")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Pipeline not found"
    registry.clear()


# ── D14: GET /api/checkpoints — 200 with checkpoint list ────────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_checkpoints_200(mock_list):
    mock_list.return_value = [
        {
            "file": "my-app_42.json",
            "project_name": "my-app",
            "issue_num": 42,
            "pipeline_mode": "standard",
            "failed_step_name": "Local CI Check",
        },
    ]
    client = _create_app()
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["file"] == "my-app_42.json"
    assert data[0]["failed_step_name"] == "Local CI Check"


# ── D15: GET /api/checkpoints — empty when no checkpoints ───────────────────


@patch("orchestrator.api.routes.list_checkpoints")
def test_checkpoints_empty(mock_list):
    mock_list.return_value = []
    client = _create_app()
    resp = client.get("/api/checkpoints")
    assert resp.status_code == 200
    assert resp.json() == []


# ── D16: Secret masking — secrets in project summary are masked ──────────────


@patch(
    "orchestrator.api.routes.load_project_summary",
    return_value={"api_key": "sk-ant-abc123456789XYZ"},
)
def test_secret_masking_project_summary(mock_load):
    client = _create_app(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.get("/api/projects/my-app")
    assert resp.status_code == 200
    body = resp.text
    assert "sk-ant-" not in body
    assert "[MASKED]" in body


# ── D17: Secret masking — secrets in pipeline context are masked ─────────────


def test_secret_masking_pipeline():
    from orchestrator.api import registry
    registry.clear()
    registry._active["my-app_42"] = dict(
        project_name="my-app", issue_num=42,
        branch_name="solve/issue-42", mode="standard", issue_title="Fix bug",
        design_doc="token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 here",
        git_diff="", review_report="", ai_audit_result="", ci_check_log="",
        steps=[dict(name="Sonnet Implement", status="passed", detail="ok", elapsed_sec=1.5)],
    )
    client = _create_app()
    resp = client.get("/api/pipelines/my-app_42")
    assert resp.status_code == 200
    body = resp.text
    assert "ghp_" not in body
    assert "[MASKED]" in body
    registry.clear()
