"""Unit tests for orchestrator/core_commands.py.

Tests are written before the implementation (TDD). All heavy dependencies
(OllamaProvider, run_fivebrid_pipeline, registry, etc.) are mocked so these
tests are fast and offline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.core_commands import (
    ShellResult,
    core_cancel,
    core_retry,
    core_shell,
    core_solve,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_settings(**overrides):
    defaults = dict(
        ollama_base_url="http://localhost:11434",
        reasoning_model="deepseek-r1:latest",
        solve_timeout=300,
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def _async_ollama(mock_ollama_cls):
    """Configure mock OllamaProvider class so its instance has an async close."""
    mock_ollama_cls.return_value.close = AsyncMock()
    return mock_ollama_cls.return_value


# ── T1: core_solve ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_single_issue_calls_pipeline_once(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """core_solve with one issue calls run_fivebrid_pipeline exactly once."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {42: asyncio.Event()}

    await core_solve(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={"path": "/tmp/my-app"},
        issue_nums=[42],
        solve_mode=None,
        parallel=False,
        settings=settings,
        cancel_events=cancel_events,
    )

    mock_pipeline.assert_awaited_once()


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_parallel_calls_pipeline_for_each_issue(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """core_solve with parallel=True and 2+ issues calls pipeline for each."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {1: asyncio.Event(), 2: asyncio.Event()}

    await core_solve(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_nums=[1, 2],
        solve_mode=None,
        parallel=True,
        settings=settings,
        cancel_events=cancel_events,
    )

    assert mock_pipeline.await_count == 2


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_sequential_calls_pipeline_for_each_issue(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """core_solve with parallel=False runs issues sequentially."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {1: asyncio.Event(), 2: asyncio.Event()}

    await core_solve(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_nums=[1, 2],
        solve_mode=None,
        parallel=False,
        settings=settings,
        cancel_events=cancel_events,
    )

    assert mock_pipeline.await_count == 2


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_registers_and_unregisters_pipeline(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """Pipeline is registered before run and unregistered in finally block."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {5: asyncio.Event()}

    await core_solve(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_nums=[5],
        solve_mode=None,
        parallel=False,
        settings=settings,
        cancel_events=cancel_events,
    )

    mock_registry.register.assert_called_once()
    mock_registry.unregister.assert_called_once_with("my-app", 5)


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_unregisters_even_on_pipeline_error(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """registry.unregister is called even when run_fivebrid_pipeline raises."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {7: asyncio.Event()}
    mock_pipeline.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await core_solve(
            project_name="my-app",
            project_path="/tmp/my-app",
            project_info={},
            issue_nums=[7],
            solve_mode=None,
            parallel=False,
            settings=settings,
            cancel_events=cancel_events,
        )

    mock_registry.unregister.assert_called_once_with("my-app", 7)


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_calls_progress_callback(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """progress_cb is forwarded to run_fivebrid_pipeline."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {3: asyncio.Event()}
    progress_cb = AsyncMock()

    await core_solve(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_nums=[3],
        solve_mode=None,
        parallel=False,
        settings=settings,
        cancel_events=cancel_events,
        progress_cb=progress_cb,
    )

    # run_fivebrid_pipeline should be called with the progress_cb
    args, kwargs = mock_pipeline.call_args
    assert args[5] is progress_cb


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_resolves_none_mode_to_standard(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """None mode resolves to 'standard' when building the PipelineContext."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_events = {1: asyncio.Event()}

    await core_solve(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_nums=[1],
        solve_mode=None,
        parallel=False,
        settings=settings,
        cancel_events=cancel_events,
    )

    _, kwargs = mock_ctx_cls.call_args
    assert kwargs.get("mode") == "standard"


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.PipelineContext")
async def test_core_solve_closes_ollama_on_error(
    mock_ctx_cls, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """OllamaProvider.close() is called even if pipeline raises."""
    mock_ollama_instance = AsyncMock()
    mock_ollama_cls.return_value = mock_ollama_instance
    settings = _make_settings()
    cancel_events = {9: asyncio.Event()}
    mock_pipeline.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError):
        await core_solve(
            project_name="my-app",
            project_path="/tmp/my-app",
            project_info={},
            issue_nums=[9],
            solve_mode=None,
            parallel=False,
            settings=settings,
            cancel_events=cancel_events,
        )

    mock_ollama_instance.close.assert_awaited_once()


# ── T2: core_retry ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.restore_context")
@patch("orchestrator.core_commands.load_checkpoint")
async def test_core_retry_loads_checkpoint_and_runs_pipeline(
    mock_load_cp, mock_restore, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """Loads checkpoint, calls restore_context, then run_fivebrid_pipeline."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_event = asyncio.Event()

    mock_ctx = MagicMock()
    mock_ctx.project_name = "my-app"
    mock_ctx.issue_num = 42
    mock_restore.return_value = mock_ctx

    mock_load_cp.return_value = {
        "ctx": {"project_name": "my-app", "issue_num": 42, "steps": []},
        "pipeline_mode": "express",
        "failed_step_index": 3,
    }

    await core_retry(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_num=42,
        settings=settings,
        cancel_event=cancel_event,
    )

    mock_load_cp.assert_called_once_with("my-app", 42)
    mock_restore.assert_called_once()
    mock_pipeline.assert_awaited_once()
    _, call_kwargs = mock_pipeline.call_args
    assert call_kwargs.get("resume_from_step") == 3


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.restore_context")
@patch("orchestrator.core_commands.load_checkpoint", return_value=None)
async def test_core_retry_missing_checkpoint_returns_early(
    mock_load_cp, mock_restore, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """If load_checkpoint returns None, function returns without calling pipeline."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_event = asyncio.Event()

    await core_retry(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_num=99,
        settings=settings,
        cancel_event=cancel_event,
    )

    mock_pipeline.assert_not_awaited()
    mock_restore.assert_not_called()


@pytest.mark.asyncio
@patch("orchestrator.core_commands.registry")
@patch("orchestrator.core_commands.OllamaProvider")
@patch("orchestrator.core_commands.GeminiCLIProvider")
@patch("orchestrator.core_commands.run_fivebrid_pipeline", new_callable=AsyncMock)
@patch("orchestrator.core_commands.restore_context")
@patch("orchestrator.core_commands.load_checkpoint")
async def test_core_retry_registers_and_unregisters(
    mock_load_cp, mock_restore, mock_pipeline, mock_gemini_cls, mock_ollama_cls, mock_registry,
):
    """Pipeline registry lifecycle is correct for retry."""
    _async_ollama(mock_ollama_cls)
    settings = _make_settings()
    cancel_event = asyncio.Event()

    mock_ctx = MagicMock()
    mock_ctx.project_name = "my-app"
    mock_ctx.issue_num = 10
    mock_restore.return_value = mock_ctx

    mock_load_cp.return_value = {
        "ctx": {"project_name": "my-app", "issue_num": 10, "steps": []},
        "pipeline_mode": "standard",
        "failed_step_index": 1,
    }

    await core_retry(
        project_name="my-app",
        project_path="/tmp/my-app",
        project_info={},
        issue_num=10,
        settings=settings,
        cancel_event=cancel_event,
    )

    mock_registry.register.assert_called_once_with(mock_ctx)
    mock_registry.unregister.assert_called_once_with("my-app", 10)


# ── T3: core_cancel ───────────────────────────────────────────────────────────


def test_core_cancel_sets_event():
    """Given a cancel_events dict with the pipeline_id, the event is set."""
    event = asyncio.Event()
    cancel_events = {"my-app_42": event}

    result = core_cancel("my-app_42", cancel_events)

    assert result is True
    assert event.is_set()


def test_core_cancel_unknown_pipeline_returns_false():
    """Returns False if pipeline_id not in cancel_events."""
    result = core_cancel("ghost_99", {})

    assert result is False


def test_core_cancel_does_not_raise_on_empty_dict():
    """core_cancel is safe with an empty dict."""
    result = core_cancel("anything", {})
    assert result is False


# ── T4: core_shell ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("orchestrator.core_commands.mask_secrets", side_effect=lambda x: x)
async def test_core_shell_runs_subprocess(mock_mask):
    """Executes command via subprocess and returns output."""
    result = await core_shell("echo hello", timeout=10)

    assert isinstance(result, ShellResult)
    assert "hello" in result.output
    assert result.timed_out is False


@pytest.mark.asyncio
@patch("orchestrator.core_commands.mask_secrets")
async def test_core_shell_masks_secrets(mock_mask):
    """Output is passed through mask_secrets before returning."""
    mock_mask.return_value = "[MASKED OUTPUT]"

    result = await core_shell("echo secret-token", timeout=10)

    mock_mask.assert_called_once()
    assert result.output == "[MASKED OUTPUT]"


@pytest.mark.asyncio
@patch("orchestrator.core_commands.mask_secrets", side_effect=lambda x: x)
async def test_core_shell_timeout_returns_timeout_result(mock_mask):
    """TimeoutError is caught and returned as a timeout result."""
    result = await core_shell("sleep 100", timeout=0)

    assert result.timed_out is True
    assert result.exit_code == -1


@pytest.mark.asyncio
@patch("orchestrator.core_commands.mask_secrets", side_effect=lambda x: x)
async def test_core_shell_returns_exit_code(mock_mask):
    """Result includes process return code."""
    result = await core_shell("exit 0", timeout=10)

    assert isinstance(result.exit_code, int)
    assert result.timed_out is False


@pytest.mark.asyncio
@patch("orchestrator.core_commands.mask_secrets", side_effect=lambda x: x)
async def test_core_shell_nonzero_exit_code(mock_mask):
    """Non-zero exit codes are captured in the result."""
    result = await core_shell("false", timeout=10)

    assert result.exit_code != 0
    assert result.timed_out is False
