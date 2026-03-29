"""Tests for project endpoints: list, detail, issues."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.app import create_api_app
from orchestrator.config import Settings


_TEST_API_KEY = "test-key-12345"


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        telegram_bot_token="fake",
        telegram_allowed_user_id=1,
        dashboard_api_key=_TEST_API_KEY,
        cors_origins="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _create_client(projects=None):
    app = create_api_app(_make_settings(), projects=projects or {})
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {_TEST_API_KEY}"})
    return c


# ── P1: GET /api/projects — returns project list ──────────────────────────────


def test_list_projects_returns_all():
    client = _create_client(projects={
        "proj-a": {"path": "/tmp/a"},
        "proj-b": {"path": "/tmp/b"},
    })
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    names = {p["name"] for p in data}
    assert names == {"proj-a", "proj-b"}


# ── P2: GET /api/projects — empty list when no projects ──────────────────────


def test_list_projects_empty():
    client = _create_client(projects={})
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == []


# ── P3: GET /api/projects/{name} — returns project detail with summary ───────


@patch("orchestrator.api.routes.load_project_summary")
def test_project_detail_with_summary(mock_summary):
    mock_summary.return_value = {"issues": [{"num": 1, "title": "Init"}]}
    client = _create_client(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.get("/api/projects/my-app")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "my-app"
    assert body["path"] == "/tmp/my-app"
    assert body["summary"]["issues"][0]["num"] == 1


# ── P4: GET /api/projects/{name} — 404 for unknown project ──────────────────


def test_project_detail_not_found():
    client = _create_client(projects={})
    resp = client.get("/api/projects/unknown")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Project not found"


# ── P5: GET /api/projects/{name} — summary is empty dict when missing ────────


@patch("orchestrator.api.routes.load_project_summary")
def test_project_detail_no_summary(mock_summary):
    mock_summary.return_value = {}
    client = _create_client(projects={"bare": {"path": "/tmp/bare"}})
    resp = client.get("/api/projects/bare")
    assert resp.status_code == 200
    assert resp.json()["summary"] == {}


# ── P6: GET /api/projects/{name}/issues — returns GitHub issues ──────────────


@patch("orchestrator.api.routes._fetch_github_issues", new_callable=AsyncMock)
def test_project_issues_success(mock_fetch):
    mock_fetch.return_value = [
        {"number": 1, "title": "Bug", "url": "https://github.com/x/y/issues/1", "labels": ["bug"]},
    ]
    client = _create_client(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.get("/api/projects/my-app/issues")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["number"] == 1
    assert data[0]["title"] == "Bug"


# ── P7: GET /api/projects/{name}/issues — 404 for unknown project ────────────


def test_project_issues_not_found():
    client = _create_client(projects={})
    resp = client.get("/api/projects/unknown/issues")
    assert resp.status_code == 404


# ── P8: GET /api/projects/{name}/issues — gh failure returns empty list ──────


@patch("orchestrator.api.routes._fetch_github_issues", new_callable=AsyncMock)
def test_project_issues_gh_failure(mock_fetch):
    mock_fetch.return_value = []
    client = _create_client(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.get("/api/projects/my-app/issues")
    assert resp.status_code == 200
    assert resp.json() == []


# ── P9: Secret masking applied to project list ───────────────────────────────


def test_list_projects_masks_secrets():
    client = _create_client(projects={
        "secret-proj": {"path": "/tmp/sk-ant-ABCDEFGHIJ"},
    })
    resp = client.get("/api/projects")
    assert "sk-ant-" not in resp.text
    assert "[MASKED]" in resp.text


# ── P10: Secret masking applied to project detail summary ────────────────────


@patch("orchestrator.api.routes.load_project_summary")
def test_project_detail_masks_secrets(mock_summary):
    mock_summary.return_value = {
        "issues": [{"num": 1, "title": "Has ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA token"}],
    }
    client = _create_client(projects={"app": {"path": "/tmp/app"}})
    resp = client.get("/api/projects/app")
    assert "ghp_" not in resp.text
    assert "[MASKED]" in resp.text


# ── P11: Secret masking applied to issues ────────────────────────────────────


@patch("orchestrator.api.routes._fetch_github_issues", new_callable=AsyncMock)
def test_project_issues_masks_secrets(mock_fetch):
    mock_fetch.return_value = [
        {
            "number": 1,
            "title": "Has ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA token",
            "url": "https://github.com/x/y/issues/1",
            "labels": [],
        },
    ]
    client = _create_client(projects={"my-app": {"path": "/tmp/my-app"}})
    resp = client.get("/api/projects/my-app/issues")
    assert "ghp_" not in resp.text
    assert "[MASKED]" in resp.text


# ── P12: Auth required ─────────────────────────────────────────────────────


def test_projects_requires_auth():
    app = create_api_app(_make_settings())
    client = TestClient(app)
    resp = client.get("/api/projects")
    assert resp.status_code == 401
