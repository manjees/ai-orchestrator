"""Tests for GET /api/pipelines endpoints."""

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


@pytest.fixture(autouse=True)
def _clear_registry():
    from orchestrator.api.registry import clear
    clear()
    yield
    clear()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _active_pipeline(
    project_name="myproject",
    issue_num=42,
    mode="standard",
    issue_title="Fix bug",
    branch_name="solve/issue-42",
    steps=None,
    **extra,
):
    """Build a ctx dict that mirrors dataclasses.asdict(PipelineContext)."""
    if steps is None:
        steps = [
            {"name": "Opus Design", "status": "passed", "detail": "", "elapsed_sec": 5.0},
            {"name": "Sonnet Implement", "status": "running", "detail": "", "elapsed_sec": 3.0},
        ]
    ctx = dict(
        project_path="/tmp/test",
        project_name=project_name,
        issue_num=issue_num,
        branch_name=branch_name,
        issue_body="",
        issue_title=issue_title,
        base_commit="abc123",
        design_doc="",
        qwen_hints="",
        git_diff="",
        review_report="",
        audit_result="",
        audit_passed=False,
        data_mining_result="",
        retry_count=0,
        review_feedback="",
        review_passed=False,
        research_log="",
        gemini_design_critique="",
        design_iteration=0,
        self_review_report="",
        gemini_cross_review="",
        impl_snapshot_ref="",
        ci_check_log="",
        ai_audit_result="",
        ai_audit_passed=False,
        ci_fix_history=[],
        audit_fix_history=[],
        mode=mode,
        triage_reason="",
        split_plan="",
        steps=steps,
        supreme_court_ruling="",
        draft_context_diff="",
        predecessor_issue_num=0,
    )
    ctx.update(extra)
    return ctx


def _checkpoint_meta(project_name="cpproject", issue_num=99):
    return {
        "file": f"{project_name}_{issue_num}.json",
        "project_name": project_name,
        "issue_num": issue_num,
        "pipeline_mode": "standard",
        "failed_step_name": "Sonnet Implement",
    }


def _checkpoint_data(project_name="cpproject", issue_num=99, steps=None):
    if steps is None:
        steps = [
            {"name": "Opus Design", "status": "passed", "detail": "", "elapsed_sec": 5.0},
            {"name": "Sonnet Implement", "status": "failed", "detail": "error", "elapsed_sec": 10.0},
        ]
    return {
        "pipeline_mode": "standard",
        "failed_step_name": "Sonnet Implement",
        "failed_step_index": 1,
        "original_project_path": "/tmp/test",
        "ctx": _active_pipeline(
            project_name=project_name,
            issue_num=issue_num,
            steps=steps,
        ),
    }


# ── GET /api/pipelines ──────────────────────────────────────────────────────


def test_list_pipelines_empty(client, auth_header):
    """T1: No active pipelines, no checkpoints → 200, empty list."""
    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)
    assert resp.status_code == 200
    assert resp.json() == {"pipelines": []}


def test_list_pipelines_with_active(client, auth_header):
    """T2: Active pipeline in registry → appears in response."""
    from orchestrator.api import registry
    ctx_data = _active_pipeline()
    registry._active["myproject_42"] = ctx_data

    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pipelines"]) == 1
    p = data["pipelines"][0]
    assert p["pipeline_id"] == "myproject_42"
    assert p["project_name"] == "myproject"
    assert p["issue_num"] == 42
    assert p["status"] == "running"
    assert p["mode"] == "standard"
    assert len(p["steps"]) == 2


def test_list_pipelines_with_completed_checkpoint(client, auth_header):
    """T3: Checkpoint pipeline appears with status derived from steps."""
    cp_meta = _checkpoint_meta()
    cp_data = _checkpoint_data()

    with patch("orchestrator.api.routes.list_checkpoints", return_value=[cp_meta]), \
         patch("orchestrator.api.routes.load_checkpoint", return_value=cp_data):
        resp = client.get("/api/pipelines", headers=auth_header)

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pipelines"]) == 1
    p = data["pipelines"][0]
    assert p["pipeline_id"] == "cpproject_99"
    assert p["status"] == "failed"  # derived from step data


def test_list_pipelines_mixed(client, auth_header):
    """T4: 1 active + 1 checkpoint → both appear."""
    from orchestrator.api import registry
    registry._active["myproject_42"] = _active_pipeline()

    cp_meta = _checkpoint_meta()
    cp_data = _checkpoint_data()

    with patch("orchestrator.api.routes.list_checkpoints", return_value=[cp_meta]), \
         patch("orchestrator.api.routes.load_checkpoint", return_value=cp_data):
        resp = client.get("/api/pipelines", headers=auth_header)

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pipelines"]) == 2
    ids = {p["pipeline_id"] for p in data["pipelines"]}
    assert ids == {"myproject_42", "cpproject_99"}


def test_list_pipelines_secret_masking(client, auth_header):
    """T5: Secret in step detail → [MASKED] in response."""
    from orchestrator.api import registry
    ctx_data = _active_pipeline(
        steps=[
            {"name": "Opus Design", "status": "passed", "detail": "key=sk-ant-abc123456789", "elapsed_sec": 1.0},
        ],
    )
    registry._active["myproject_42"] = ctx_data

    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)

    assert resp.status_code == 200
    p = resp.json()["pipelines"][0]
    assert "sk-ant-" not in p["steps"][0]["detail"]
    assert "[MASKED]" in p["steps"][0]["detail"]


def test_list_pipelines_requires_auth(client):
    """T6: No auth header → 401."""
    resp = client.get("/api/pipelines")
    assert resp.status_code == 401


# ── GET /api/pipelines/{pipeline_id} ────────────────────────────────────────


def test_get_pipeline_detail_active(client, auth_header):
    """T7: Active pipeline → 200, full detail."""
    from orchestrator.api import registry
    ctx_data = _active_pipeline(design_doc="some design", research_log="did research")
    registry._active["myproject_42"] = ctx_data

    resp = client.get("/api/pipelines/myproject_42", headers=auth_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_id"] == "myproject_42"
    assert data["design_doc"] == "some design"
    assert data["research_log"] == "did research"
    assert data["status"] == "running"


def test_get_pipeline_detail_from_checkpoint(client, auth_header):
    """T8: No active, checkpoint exists → 200."""
    cp_data = _checkpoint_data()

    with patch("orchestrator.api.routes.load_checkpoint", return_value=cp_data):
        resp = client.get("/api/pipelines/cpproject_99", headers=auth_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline_id"] == "cpproject_99"
    assert data["status"] == "failed"


def test_get_pipeline_not_found(client, auth_header):
    """T9: No active, no checkpoint → 404."""
    with patch("orchestrator.api.routes.load_checkpoint", return_value=None):
        resp = client.get("/api/pipelines/nonexistent_1", headers=auth_header)

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Pipeline not found"}


def test_get_pipeline_detail_secret_masking(client, auth_header):
    """T10: Secrets in string fields → all masked."""
    from orchestrator.api import registry
    ctx_data = _active_pipeline(
        design_doc="token: sk-ant-secret1234567890",
        review_report="key ghp_abcdefghij1234567890abcdefghij123456",
    )
    registry._active["myproject_42"] = ctx_data

    resp = client.get("/api/pipelines/myproject_42", headers=auth_header)

    assert resp.status_code == 200
    data = resp.json()
    assert "sk-ant-" not in data["design_doc"]
    assert "[MASKED]" in data["design_doc"]
    assert "ghp_" not in data["review_report"]
    assert "[MASKED]" in data["review_report"]


def test_get_pipeline_detail_requires_auth(client):
    """T11: No auth header → 401."""
    resp = client.get("/api/pipelines/myproject_42")
    assert resp.status_code == 401


# ── Status derivation ────────────────────────────────────────────────────────


def test_derive_status_running(client, auth_header):
    """T12: Steps with one running → status = running."""
    from orchestrator.api import registry
    registry._active["p_1"] = _active_pipeline(
        project_name="p", issue_num=1,
        steps=[
            {"name": "A", "status": "passed", "detail": "", "elapsed_sec": 0},
            {"name": "B", "status": "running", "detail": "", "elapsed_sec": 0},
        ],
    )
    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)
    assert resp.json()["pipelines"][0]["status"] == "running"


def test_derive_status_completed(client, auth_header):
    """T13: All steps passed → status = completed."""
    from orchestrator.api import registry
    registry._active["p_1"] = _active_pipeline(
        project_name="p", issue_num=1,
        steps=[
            {"name": "A", "status": "passed", "detail": "", "elapsed_sec": 0},
            {"name": "B", "status": "skipped", "detail": "", "elapsed_sec": 0},
        ],
    )
    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)
    assert resp.json()["pipelines"][0]["status"] == "completed"


def test_derive_status_failed(client, auth_header):
    """T14: One step failed, none running → status = failed."""
    from orchestrator.api import registry
    registry._active["p_1"] = _active_pipeline(
        project_name="p", issue_num=1,
        steps=[
            {"name": "A", "status": "passed", "detail": "", "elapsed_sec": 0},
            {"name": "B", "status": "failed", "detail": "", "elapsed_sec": 0},
        ],
    )
    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)
    assert resp.json()["pipelines"][0]["status"] == "failed"


def test_derive_status_pending(client, auth_header):
    """T15: All steps pending → status = pending."""
    from orchestrator.api import registry
    registry._active["p_1"] = _active_pipeline(
        project_name="p", issue_num=1,
        steps=[
            {"name": "A", "status": "pending", "detail": "", "elapsed_sec": 0},
            {"name": "B", "status": "pending", "detail": "", "elapsed_sec": 0},
        ],
    )
    with patch("orchestrator.api.routes.list_checkpoints", return_value=[]):
        resp = client.get("/api/pipelines", headers=auth_header)
    assert resp.json()["pipelines"][0]["status"] == "pending"
