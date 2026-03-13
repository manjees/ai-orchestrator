"""Tests for pipeline event hooks — dashboard integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.pipeline import (
    PipelineStep,
    _STEP_AGENT_MAP,
    _notify_step,
    run_fivebrid_pipeline,
    PipelineContext,
)


@pytest.fixture
def mock_dashboard():
    client = AsyncMock()
    client.update_agent_status = AsyncMock()
    client.send_event = AsyncMock()
    client.create_pipeline_run = AsyncMock(return_value="run-123")
    client.update_pipeline_run = AsyncMock()
    return client


def _make_ctx(**overrides):
    """Build a minimal PipelineContext for testing."""
    defaults = dict(
        project_path="/tmp/test",
        project_name="test-project",
        issue_num=42,
        branch_name="solve/issue-42",
        mode="standard",
        issue_body="test body",
        issue_title="test title",
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


def _pipeline_patches(extra_step_patches=None):
    """Common patches for running the pipeline in tests."""
    import contextlib

    patches = [
        patch("orchestrator.pipeline.step_fetch_issue", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_haiku_research", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_opus_design", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_gemini_design_critique", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_qwen_pre_implement", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_claude_implement", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_sonnet_self_review", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_gemini_cross_review", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_ai_audit", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_data_mining_fivebrid", new_callable=AsyncMock),
        patch("orchestrator.pipeline.step_local_ci_check", new_callable=AsyncMock),
        patch("orchestrator.pipeline._capture_filtered_diff", new_callable=AsyncMock, return_value="diff"),
        patch("orchestrator.pipeline.build_fivebrid_steps"),
        patch("orchestrator.pipeline._resolve_ci_commands", return_value=[]),
        patch("orchestrator.state_sync.format_state_context", return_value=""),
        patch("orchestrator.state_sync.append_project_summary"),
        patch("orchestrator.pipeline._extract_decisions_llm", new_callable=AsyncMock, return_value=""),
        patch("orchestrator.pipeline._extract_files_from_diff", return_value=[]),
        patch("orchestrator.pipeline.PIPELINE_MODES", {
            "standard": {
                "max_design_retries": 0,
                "ai_audit_max_retries": 0,
                "ai_audit_enabled": False,
                "local_ci_fix_retries": 0,
            },
        }),
    ]
    if extra_step_patches:
        patches.extend(extra_step_patches)
    return contextlib.ExitStack(), patches


async def _run_pipeline_with_mocks(ctx, mock_dashboard=None, resume_from_step=-1, build_steps=None):
    """Helper to run the pipeline with all steps mocked."""
    stack, patches = _pipeline_patches()
    cancel = asyncio.Event()
    progress = AsyncMock()
    settings = MagicMock()
    settings.local_ci_enabled = False
    settings.enable_data_mining = False

    with stack:
        mocks = {}
        for p in patches:
            m = stack.enter_context(p)
            # Track named mocks
            attr = getattr(p, 'attribute', None)
            if attr:
                mocks[attr] = m

        # Setup build_fivebrid_steps
        if build_steps is None:
            build_steps = [
                PipelineStep(name="Haiku Research"),
                PipelineStep(name="Opus Design"),
                PipelineStep(name="Gemini Design Critique"),
                PipelineStep(name="Sonnet Implement"),
            ]
        mocks.get("build_fivebrid_steps", MagicMock()).return_value = build_steps

        # Make step_claude_implement set git_diff
        impl_mock = mocks.get("step_claude_implement")
        if impl_mock:
            async def fake_implement(ctx, *a, **kw):
                ctx.git_diff = "some diff"
            impl_mock.side_effect = fake_implement

        # Mock subprocess for git rev-parse
        with patch("orchestrator.pipeline.asyncio.create_subprocess_exec") as mock_proc:
            proc_mock = AsyncMock()
            proc_mock.communicate = AsyncMock(return_value=(b"abc123", b""))
            mock_proc.return_value = proc_mock

            status, detail = await run_fivebrid_pipeline(
                ctx, AsyncMock(), AsyncMock(), settings, cancel, progress,
                dashboard_client=mock_dashboard,
                resume_from_step=resume_from_step,
            )

    return status, detail


@pytest.mark.asyncio
async def test_notify_step_calls_update_agent_status(mock_dashboard):
    await _notify_step(mock_dashboard, "Haiku Research", "RUNNING")
    mock_dashboard.update_agent_status.assert_awaited_once_with("haiku-researcher", "RUNNING")


@pytest.mark.asyncio
async def test_notify_step_noop_when_client_is_none():
    await _notify_step(None, "Haiku Research", "RUNNING")


@pytest.mark.asyncio
async def test_notify_step_noop_for_unknown_step(mock_dashboard):
    await _notify_step(mock_dashboard, "Unknown Step", "RUNNING")
    mock_dashboard.update_agent_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_notify_step_all_mapped_steps(mock_dashboard):
    """Verify every entry in _STEP_AGENT_MAP is callable via _notify_step."""
    for step_name, agent_id in _STEP_AGENT_MAP.items():
        mock_dashboard.reset_mock()
        await _notify_step(mock_dashboard, step_name, "RUNNING")
        mock_dashboard.update_agent_status.assert_awaited_once_with(agent_id, "RUNNING")


@pytest.mark.asyncio
async def test_pipeline_started_event(mock_dashboard):
    """PIPELINE_STARTED event sent at pipeline start."""
    ctx = _make_ctx()
    status, _ = await _run_pipeline_with_mocks(ctx, mock_dashboard=mock_dashboard)

    calls = [c for c in mock_dashboard.send_event.call_args_list if c.args[0] == "PIPELINE_STARTED"]
    assert len(calls) == 1
    event_data = calls[0].args[1]
    assert event_data["issue_num"] == 42
    assert event_data["mode"] == "standard"
    assert event_data["project_name"] == "test-project"


@pytest.mark.asyncio
async def test_pipeline_completed_event(mock_dashboard):
    """PIPELINE_COMPLETED event sent on success."""
    ctx = _make_ctx()
    status, _ = await _run_pipeline_with_mocks(ctx, mock_dashboard=mock_dashboard)

    assert status == "success"
    calls = [c for c in mock_dashboard.send_event.call_args_list if c.args[0] == "PIPELINE_COMPLETED"]
    assert len(calls) == 1
    assert calls[0].args[1]["status"] == "success"


@pytest.mark.asyncio
async def test_error_event_on_exception(mock_dashboard):
    """ERROR event sent on pipeline exception."""
    ctx = _make_ctx()
    cancel = asyncio.Event()
    progress = AsyncMock()
    settings = MagicMock()
    settings.local_ci_enabled = False
    settings.enable_data_mining = False

    with patch("orchestrator.pipeline.step_fetch_issue", new_callable=AsyncMock), \
         patch("orchestrator.pipeline.step_haiku_research", side_effect=RuntimeError("boom")), \
         patch("orchestrator.pipeline.build_fivebrid_steps") as mock_build, \
         patch("orchestrator.pipeline._resolve_ci_commands", return_value=[]), \
         patch("orchestrator.state_sync.format_state_context", return_value=""), \
         patch("orchestrator.pipeline.PIPELINE_MODES", {"standard": {"max_design_retries": 0, "ai_audit_max_retries": 0, "ai_audit_enabled": False, "local_ci_fix_retries": 0}}):

        mock_build.return_value = [
            PipelineStep(name="Haiku Research"),
            PipelineStep(name="Opus Design"),
            PipelineStep(name="Sonnet Implement"),
        ]

        status, _ = await run_fivebrid_pipeline(
            ctx, AsyncMock(), AsyncMock(), settings, cancel, progress,
            dashboard_client=mock_dashboard,
        )

    assert status == "failed"
    calls = [c for c in mock_dashboard.send_event.call_args_list if c.args[0] == "ERROR"]
    assert len(calls) == 1
    assert "error_detail" in calls[0].args[1]


@pytest.mark.asyncio
async def test_no_events_when_dashboard_is_none():
    """Backward compat: no crash when dashboard_client is None."""
    ctx = _make_ctx()
    status, _ = await _run_pipeline_with_mocks(ctx, mock_dashboard=None)
    assert status == "success"


@pytest.mark.asyncio
async def test_steps_send_running_and_idle(mock_dashboard):
    """Each step sends RUNNING before and IDLE after execution."""
    ctx = _make_ctx()
    await _run_pipeline_with_mocks(ctx, mock_dashboard=mock_dashboard)

    status_calls = mock_dashboard.update_agent_status.call_args_list
    agent_transitions = [(c.args[0], c.args[1]) for c in status_calls]

    # Haiku Research
    assert ("haiku-researcher", "RUNNING") in agent_transitions
    assert ("haiku-researcher", "IDLE") in agent_transitions

    # Opus Design
    assert ("opus-designer", "RUNNING") in agent_transitions
    assert ("opus-designer", "IDLE") in agent_transitions

    # Gemini Design Critique
    assert ("gemini-reviewer", "RUNNING") in agent_transitions
    assert ("gemini-reviewer", "IDLE") in agent_transitions

    # Sonnet Implement
    assert ("sonnet-worker", "RUNNING") in agent_transitions
    assert ("sonnet-worker", "IDLE") in agent_transitions

    haiku_running_idx = agent_transitions.index(("haiku-researcher", "RUNNING"))
    haiku_idle_idx = agent_transitions.index(("haiku-researcher", "IDLE"))
    assert haiku_running_idx < haiku_idle_idx


@pytest.mark.asyncio
async def test_cached_steps_skip_running_idle(mock_dashboard):
    """Cached/resumed steps do NOT send RUNNING/IDLE."""
    ctx = _make_ctx()
    ctx.steps = [
        PipelineStep(name="Haiku Research", status="passed"),
        PipelineStep(name="Opus Design", status="passed"),
        PipelineStep(name="Gemini Design Critique", status="passed"),
        PipelineStep(name="Sonnet Implement"),
    ]

    await _run_pipeline_with_mocks(ctx, mock_dashboard=mock_dashboard, resume_from_step=3)

    status_calls = mock_dashboard.update_agent_status.call_args_list
    agent_transitions = [(c.args[0], c.args[1]) for c in status_calls]

    # Haiku, Opus, Gemini Critique should NOT have RUNNING/IDLE (cached)
    assert ("haiku-researcher", "RUNNING") not in agent_transitions
    assert ("opus-designer", "RUNNING") not in agent_transitions

    # Sonnet Implement should still have RUNNING/IDLE (not cached)
    assert ("sonnet-worker", "RUNNING") in agent_transitions
    assert ("sonnet-worker", "IDLE") in agent_transitions


@pytest.mark.asyncio
async def test_solve_with_fivebrid_passes_dashboard_client(mock_dashboard):
    """_solve_with_fivebrid passes dashboard_client to run_fivebrid_pipeline."""
    from orchestrator.handlers import _solve_with_fivebrid

    context = MagicMock()
    context.bot = AsyncMock()
    context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    context.bot.edit_message_text = AsyncMock()
    settings = MagicMock()
    settings.solve_mode = "auto"
    context.bot_data = {"settings": settings, "dashboard": mock_dashboard}

    with patch("orchestrator.handlers._git_fresh_start", new_callable=AsyncMock, return_value=(True, "/tmp/wt")), \
         patch("orchestrator.handlers.step_triage_and_split", new_callable=AsyncMock, return_value={"mode": "standard", "split_needed": False, "reason": "ok", "sub_issues": [], "estimated_files": ""}), \
         patch("orchestrator.handlers.run_fivebrid_pipeline", new_callable=AsyncMock, return_value=("success", "done")) as mock_run, \
         patch("orchestrator.handlers._create_pr", new_callable=AsyncMock, return_value="http://pr"), \
         patch("orchestrator.handlers.format_pipeline_summary", return_value="summary"), \
         patch("orchestrator.handlers.get_system_status", new_callable=AsyncMock), \
         patch("orchestrator.handlers.PIPELINE_MODES", {"standard": {"description": "Standard", "label": "std", "estimated_minutes": (5, 10)}}), \
         patch("orchestrator.handlers.asyncio.create_subprocess_exec") as mock_proc:

        proc_mock = AsyncMock()
        proc_mock.communicate = AsyncMock(return_value=(b"abc123", b""))
        mock_proc.return_value = proc_mock

        fivebrid_settings = MagicMock()
        fivebrid_settings.solve_mode = "auto"
        fivebrid_settings.strategy_approval_timeout = 300

        await _solve_with_fivebrid(
            context, 1, "proj", "/tmp/proj", 42, 600,
            asyncio.Event(), AsyncMock(), AsyncMock(), fivebrid_settings,
            dashboard_client=mock_dashboard,
        )

        mock_run.assert_awaited_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("dashboard_client") is mock_dashboard


@pytest.mark.asyncio
async def test_solve_single_issue_passes_dashboard_client(mock_dashboard):
    """_solve_single_issue passes dashboard_client to _solve_with_fivebrid."""
    from orchestrator.handlers import _solve_single_issue

    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(),
        "projects": {"proj": {}},
        "ollama": AsyncMock(),
        "gemini": AsyncMock(),
        "dashboard": mock_dashboard,
    }

    with patch("orchestrator.handlers._solve_with_fivebrid", new_callable=AsyncMock, return_value=("success", "done")) as mock_solve:
        await _solve_single_issue(
            context, 1, "proj", "/tmp/proj", 42, 600,
            asyncio.Event(),
            dashboard_client=mock_dashboard,
        )

        mock_solve.assert_awaited_once()
        call_kwargs = mock_solve.call_args.kwargs
        assert call_kwargs.get("dashboard_client") is mock_dashboard


@pytest.mark.asyncio
async def test_no_crash_when_bot_data_has_no_dashboard():
    """When bot_data has no 'dashboard' key, None is passed (no crash)."""
    from orchestrator.handlers import _solve_single_issue

    context = MagicMock()
    context.bot_data = {
        "settings": MagicMock(),
        "projects": {"proj": {}},
        "ollama": AsyncMock(),
        "gemini": AsyncMock(),
    }

    with patch("orchestrator.handlers._solve_with_fivebrid", new_callable=AsyncMock, return_value=("success", "done")) as mock_solve:
        await _solve_single_issue(
            context, 1, "proj", "/tmp/proj", 42, 600,
            asyncio.Event(),
        )

        mock_solve.assert_awaited_once()
        call_kwargs = mock_solve.call_args.kwargs
        assert call_kwargs.get("dashboard_client") is None
