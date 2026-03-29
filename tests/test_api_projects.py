"""Tests for project endpoints: list, detail, issues."""

import asyncio
import json
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


@pytest.fixture()
def client():
    app = create_api_app(_make_settings())
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {_TEST_API_KEY}"})
    return c


# ── P1: GET /api/projects — returns project list ──────────────────────────────


def test_list_projects_returns_all(client):
    with patch("orchestrator.api.routes.load_projects") as m:
        m.return_value = {
            "proj-a": {"path": "/tmp/a"},
            "proj-b": {"path": "/tmp/b"},
        }
        resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == {
        "projects": [
            {"name": "proj-a", "path": "/tmp/a"},
            {"name": "proj-b", "path": "/tmp/b"},
        ]
    }


# ── P2: GET /api/projects — empty list when no projects ──────────────────────


def test_list_projects_empty(client):
    with patch("orchestrator.api.routes.load_projects") as m:
        m.return_value = {}
        resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


# ── P3: GET /api/projects/{name} — returns project detail with summary ───────


def test_project_detail_with_summary(client):
    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.load_project_summary") as ms,
    ):
        mp.return_value = {"my-app": {"path": "/tmp/my-app"}}
        ms.return_value = {"issues": [{"num": 1, "title": "Init"}]}
        resp = client.get("/api/projects/my-app")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "my-app"
    assert body["path"] == "/tmp/my-app"
    assert body["summary"]["issues"][0]["num"] == 1


# ── P4: GET /api/projects/{name} — 404 for unknown project ──────────────────


def test_project_detail_not_found(client):
    with patch("orchestrator.api.routes.load_projects") as m:
        m.return_value = {}
        resp = client.get("/api/projects/unknown")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Project not found: unknown"


# ── P5: GET /api/projects/{name} — summary is empty dict when missing ────────


def test_project_detail_no_summary(client):
    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.load_project_summary") as ms,
    ):
        mp.return_value = {"bare": {"path": "/tmp/bare"}}
        ms.return_value = {}
        resp = client.get("/api/projects/bare")
    assert resp.status_code == 200
    assert resp.json()["summary"] == {}


# ── P6: GET /api/projects/{name}/issues — returns GitHub issues ──────────────


def test_project_issues_success(client):
    issues_json = json.dumps([
        {"number": 1, "title": "Bug", "url": "https://github.com/x/y/issues/1", "body": "desc"},
    ]).encode()

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (issues_json, b"")
    mock_proc.returncode = 0

    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.asyncio.create_subprocess_exec", return_value=mock_proc) as msub,
    ):
        mp.return_value = {"my-app": {"path": "/tmp/my-app"}}
        resp = client.get("/api/projects/my-app/issues")

    assert resp.status_code == 200
    body = resp.json()
    assert body["project"] == "my-app"
    assert body["issues"][0]["number"] == 1
    assert body["issues"][0]["title"] == "Bug"


# ── P7: GET /api/projects/{name}/issues — 404 for unknown project ────────────


def test_project_issues_not_found(client):
    with patch("orchestrator.api.routes.load_projects") as m:
        m.return_value = {}
        resp = client.get("/api/projects/unknown/issues")
    assert resp.status_code == 404


# ── P8: GET /api/projects/{name}/issues — gh command failure returns 502 ─────


def test_project_issues_gh_error(client):
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"gh: not logged in")
    mock_proc.returncode = 1

    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        mp.return_value = {"my-app": {"path": "/tmp/my-app"}}
        resp = client.get("/api/projects/my-app/issues")

    assert resp.status_code == 502
    assert "gh issue list failed" in resp.json()["detail"]


# ── P9: Secret masking applied to project list ───────────────────────────────


def test_list_projects_masks_secrets(client):
    with patch("orchestrator.api.routes.load_projects") as m:
        m.return_value = {
            "secret-proj": {"path": "/tmp/sk-ant-ABCDEFGHIJ"},
        }
        resp = client.get("/api/projects")
    assert "sk-ant-" not in resp.text
    assert "[MASKED]" in resp.text


# ── P10: Secret masking applied to project detail summary ────────────────────


def test_project_detail_masks_secrets(client):
    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.load_project_summary") as ms,
    ):
        mp.return_value = {"app": {"path": "/tmp/app"}}
        ms.return_value = {
            "issues": [{"num": 1, "title": "Has ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA token"}],
        }
        resp = client.get("/api/projects/app")
    assert "ghp_" not in resp.text
    assert "[MASKED]" in resp.text


# ── P11: Secret masking applied to issues ────────────────────────────────────


def test_project_issues_masks_secrets(client):
    issues_json = json.dumps([
        {
            "number": 1,
            "title": "Fix",
            "body": "token ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA leaked",
            "url": "https://github.com/x/y/issues/1",
        },
    ]).encode()

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (issues_json, b"")
    mock_proc.returncode = 0

    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        mp.return_value = {"my-app": {"path": "/tmp/my-app"}}
        resp = client.get("/api/projects/my-app/issues")

    assert "ghp_" not in resp.text
    assert "[MASKED]" in resp.text


# ── P12: GET /api/projects/{name}/issues — timeout returns 504 ───────────────


def test_project_issues_gh_not_found(client):
    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.asyncio.create_subprocess_exec", side_effect=FileNotFoundError),
    ):
        mp.return_value = {"my-app": {"path": "/tmp/my-app"}}
        resp = client.get("/api/projects/my-app/issues")

    assert resp.status_code == 502
    assert "gh not found" in resp.json()["detail"]


def test_project_issues_timeout(client):
    with (
        patch("orchestrator.api.routes.load_projects") as mp,
        patch("orchestrator.api.routes.asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError),
    ):
        mp.return_value = {"my-app": {"path": "/tmp/my-app"}}
        resp = client.get("/api/projects/my-app/issues")

    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]
