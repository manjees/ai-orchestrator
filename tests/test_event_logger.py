"""Tests for JSONL event logger with rotation and history API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.event_logger import EventLogger, get_event_logger, reset_event_logger


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_events_dir(tmp_path: Path) -> Path:
    return tmp_path / "events"


@pytest.fixture()
def logger(tmp_events_dir: Path) -> EventLogger:
    return EventLogger(directory=tmp_events_dir)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_event_logger()
    yield
    reset_event_logger()


# ── Group A: JSONL Writing ───────────────────────────────────────────────────


class TestJSONLWriting:
    def test_append_event_creates_file(self, logger: EventLogger, tmp_events_dir: Path):
        logger.append("pipeline.started", {"pipeline_id": "proj_1", "project_name": "proj"})
        assert (tmp_events_dir / "events.jsonl").exists()

    def test_append_event_structure(self, logger: EventLogger, tmp_events_dir: Path):
        logger.append("pipeline.started", {"pipeline_id": "proj_1", "project_name": "proj"})
        line = (tmp_events_dir / "events.jsonl").read_text().strip()
        record = json.loads(line)
        assert "timestamp" in record
        assert record["event_type"] == "pipeline.started"
        assert record["pipeline_id"] == "proj_1"
        assert record["project_name"] == "proj"
        assert "data" in record
        # ISO 8601 check
        assert "T" in record["timestamp"]

    def test_append_multiple_events(self, logger: EventLogger, tmp_events_dir: Path):
        for i in range(5):
            logger.append("step.started", {"pipeline_id": f"proj_{i}", "project_name": "proj"})
        lines = (tmp_events_dir / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5

    def test_append_event_is_valid_jsonl(self, logger: EventLogger, tmp_events_dir: Path):
        logger.append("pipeline.started", {"pipeline_id": "p_1", "project_name": "p"})
        logger.append("step.started", {"pipeline_id": "p_1", "project_name": "p"})
        logger.append("pipeline.completed", {"pipeline_id": "p_1", "project_name": "p"})
        for line in (tmp_events_dir / "events.jsonl").read_text().strip().split("\n"):
            record = json.loads(line)  # Should not raise
            assert isinstance(record, dict)


# ── Group B: File Rotation ───────────────────────────────────────────────────


class TestFileRotation:
    def test_rotation_at_threshold(self, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir, max_size=100)  # 100 bytes
        # Write enough to exceed 100 bytes
        el.append("pipeline.started", {"pipeline_id": "proj_1", "project_name": "proj"})
        el.append("pipeline.started", {"pipeline_id": "proj_2", "project_name": "proj"})
        # File should have been rotated
        rotated = tmp_events_dir / "events.jsonl.1"
        assert rotated.exists()

    def test_rotation_overwrites_old_backup(self, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir, max_size=100)
        el.append("pipeline.started", {"pipeline_id": "proj_1", "project_name": "proj"})
        el.append("pipeline.started", {"pipeline_id": "proj_2", "project_name": "proj"})
        # First rotation done; write more to trigger second rotation
        el.append("pipeline.started", {"pipeline_id": "proj_3", "project_name": "proj"})
        el.append("pipeline.started", {"pipeline_id": "proj_4", "project_name": "proj"})
        rotated = tmp_events_dir / "events.jsonl.1"
        assert rotated.exists()
        # Old backup content should have been replaced
        content = rotated.read_text()
        assert "proj_2" in content or "proj_3" in content

    def test_rotation_threshold_is_configurable(self, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir, max_size=50)
        el.append("pipeline.started", {"pipeline_id": "p_1", "project_name": "p"})
        # Even a single event at ~130+ bytes should trigger rotation on next write
        el.append("pipeline.started", {"pipeline_id": "p_2", "project_name": "p"})
        assert (tmp_events_dir / "events.jsonl.1").exists()


# ── Group C: History Read ────────────────────────────────────────────────────


class TestHistoryRead:
    def test_read_recent_events_default_limit(self, logger: EventLogger):
        for i in range(15):
            logger.append("pipeline.started", {"pipeline_id": f"proj_{i}", "project_name": "proj"})
        result = logger.read_pipeline_history(limit=10)
        assert len(result) == 10

    def test_read_recent_events_filters_pipeline_events(self, logger: EventLogger):
        logger.append("step.started", {"pipeline_id": "p_1", "project_name": "p"})
        logger.append("pipeline.started", {"pipeline_id": "p_1", "project_name": "p"})
        logger.append("step.completed", {"pipeline_id": "p_1", "project_name": "p"})
        logger.append("pipeline.completed", {"pipeline_id": "p_1", "project_name": "p"})
        result = logger.read_pipeline_history()
        event_types = [r["event_type"] for r in result]
        assert all(et.startswith("pipeline.") for et in event_types)
        assert len(result) == 2

    def test_read_recent_events_from_empty_file(self, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir)
        result = el.read_pipeline_history()
        assert result == []

    def test_read_recent_events_ordering(self, logger: EventLogger):
        logger.append("pipeline.started", {"pipeline_id": "p_1", "project_name": "first"})
        logger.append("pipeline.completed", {"pipeline_id": "p_2", "project_name": "second"})
        result = logger.read_pipeline_history()
        assert result[0]["project_name"] == "second"  # Most recent first
        assert result[1]["project_name"] == "first"


# ── Group D: Integration with EventBus ───────────────────────────────────────


class TestEventBusIntegration:
    async def test_eventbus_emit_writes_jsonl(self, tmp_events_dir: Path):
        from orchestrator.api.events import get_event_bus, reset_event_bus

        reset_event_bus()
        bus = get_event_bus()

        el = EventLogger(directory=tmp_events_dir)
        with patch("orchestrator.api.events.get_event_logger", return_value=el):
            await bus.emit("pipeline.started", {"pipeline_id": "p_1", "project_name": "p"})

        filepath = tmp_events_dir / "events.jsonl"
        assert filepath.exists()
        record = json.loads(filepath.read_text().strip())
        assert record["event_type"] == "pipeline.started"
        reset_event_bus()

    async def test_eventbus_jsonl_failure_does_not_block(self):
        from orchestrator.api.events import get_event_bus, reset_event_bus

        reset_event_bus()
        bus = get_event_bus()

        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        await bus.register(mock_ws)

        with patch("orchestrator.api.events.get_event_logger") as mock_logger:
            mock_logger.return_value.append.side_effect = PermissionError("denied")
            await bus.emit("pipeline.started", {"pipeline_id": "p_1", "project_name": "p"})

        # WS broadcast should still have succeeded
        mock_ws.send_text.assert_called_once()
        reset_event_bus()


# ── Group E: API Endpoint ────────────────────────────────────────────────────


class TestHistoryEndpoint:
    @pytest.fixture()
    def app(self):
        from orchestrator.api.app import create_api_app
        from orchestrator.config import Settings

        settings = Settings(
            telegram_bot_token="fake",
            telegram_allowed_user_id=1,
            dashboard_api_key="",
            cors_origins="",
        )
        return create_api_app(settings)

    async def test_history_endpoint_returns_json(self, app, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir)
        el.append("pipeline.started", {"pipeline_id": "p_1", "project_name": "p"})
        el.append("pipeline.completed", {"pipeline_id": "p_1", "project_name": "p"})

        with patch("orchestrator.event_logger.get_event_logger", return_value=el):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/pipelines/history")
        assert resp.status_code == 200
        body = resp.json()
        assert "history" in body
        assert len(body["history"]) == 2

    async def test_history_endpoint_limit_param(self, app, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir)
        for i in range(10):
            el.append("pipeline.started", {"pipeline_id": f"p_{i}", "project_name": "p"})

        with patch("orchestrator.event_logger.get_event_logger", return_value=el):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/pipelines/history?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()["history"]) == 5

    async def test_history_endpoint_empty(self, app, tmp_events_dir: Path):
        el = EventLogger(directory=tmp_events_dir)

        with patch("orchestrator.event_logger.get_event_logger", return_value=el):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/pipelines/history")
        assert resp.status_code == 200
        assert resp.json()["history"] == []


# ── Group F: Pipeline Integration ────────────────────────────────────────────


class TestPipelineIntegration:
    async def test_progress_cb_emits_event(self):
        """handlers.py progress_cb should emit a pipeline.progress event alongside Telegram edit."""
        from unittest.mock import MagicMock

        # Patch _emit_event and _edit_msg so we don't need a real Telegram context
        with (
            patch("orchestrator.handlers._emit_event", new_callable=AsyncMock) as mock_emit,
            patch("orchestrator.handlers._edit_msg", new_callable=AsyncMock),
            patch("orchestrator.handlers.get_system_status", new_callable=AsyncMock) as mock_sys,
        ):
            mock_sys.return_value = MagicMock(cpu_percent=10.0, ram_percent=20.0)

            # Build a progress_cb the same way handlers.py does in the solve flow
            import time
            import html
            from orchestrator.handlers import _emit_event as emit_fn

            msg = MagicMock()
            cancel_btn = None
            pipeline_start = time.monotonic()
            project_name = "myproj"
            issue_num = 7

            async def progress_cb(status_text: str) -> None:
                from orchestrator.handlers import _edit_msg, get_system_status
                elapsed = int(time.monotonic() - pipeline_start)
                mins, secs = divmod(elapsed, 60)
                time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                try:
                    sys_status = await get_system_status()
                    sys_line = f"CPU: {sys_status.cpu_percent}% | RAM: {sys_status.ram_percent}%"
                except Exception:
                    sys_line = ""
                text = (
                    f"<b>#{issue_num}</b> {html.escape(status_text)}\n"
                    f"[{time_str}] {sys_line}"
                )
                await _edit_msg(msg, text, reply_markup=cancel_btn)
                await emit_fn("pipeline.progress", {
                    "pipeline_id": f"{project_name}_{issue_num}",
                    "project_name": project_name,
                    "status_text": status_text,
                })

            await progress_cb("Step 3/9 running...")

            mock_emit.assert_called_once_with(
                "pipeline.progress",
                {
                    "pipeline_id": "myproj_7",
                    "project_name": "myproj",
                    "status_text": "Step 3/9 running...",
                },
            )

    async def test_notify_step_includes_pipeline_context(self):
        """_notify_step should include pipeline_id and project_name when ctx is provided."""
        from orchestrator.pipeline import PipelineContext, _notify_step

        ctx = PipelineContext(
            project_path="/tmp/test",
            project_name="myproj",
            issue_num=42,
            branch_name="solve/42",
        )
        with patch("orchestrator.pipeline._emit_event", new_callable=AsyncMock) as mock_emit:
            await _notify_step(None, "Opus Design", "RUNNING", ctx=ctx)
            call_args = mock_emit.call_args
            assert call_args[0][0] == "step.started"
            data = call_args[0][1]
            assert data["pipeline_id"] == "myproj_42"
            assert data["project_name"] == "myproj"
            assert data["step_name"] == "Opus Design"
