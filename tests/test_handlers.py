"""Unit tests for Telegram handler logic in orchestrator/handlers.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import approval_store
from orchestrator.handlers import supreme_court_callback, strategy_callback


@pytest.fixture(autouse=True)
def cleanup_store():
    yield
    approval_store.clear_all()


def _make_callback_query(data: str) -> MagicMock:
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


def _make_update(query: MagicMock) -> MagicMock:
    update = MagicMock()
    update.callback_query = query
    return update


def _make_context() -> MagicMock:
    return MagicMock()


# ── T1: complex key with colons is parsed correctly ──────────────────────────

@pytest.mark.asyncio
async def test_supreme_court_callback_parses_complex_key():
    """The approval key 'court:123:45' contains colons; it must not be truncated."""
    key = "court:123:45"
    approval_store.create_approval(key, "supreme_court", ["accept", "uphold", "overturn"])

    query = _make_callback_query(f"court_uphold:{key}")
    update = _make_update(query)
    context = _make_context()

    await supreme_court_callback(update, context)

    approval = approval_store.get_approval(key)
    assert approval is not None
    assert approval.decision == "uphold"
    assert approval.status == "decided"


# ── T2: approval_store.respond called with the full key ──────────────────────

@pytest.mark.asyncio
async def test_supreme_court_callback_calls_respond_with_full_key():
    key = "court:999:7"
    approval_store.create_approval(key, "supreme_court", ["accept", "uphold", "overturn"])

    query = _make_callback_query(f"court_accept:{key}")
    update = _make_update(query)

    with patch.object(approval_store, "respond", wraps=approval_store.respond) as mock_respond:
        await supreme_court_callback(update, _make_context())
        mock_respond.assert_called_once_with(key, "accept")


# ── T3: expired / missing key shows "Court session expired" ──────────────────

@pytest.mark.asyncio
async def test_supreme_court_callback_expired_key_shows_expired_message():
    query = _make_callback_query("court_uphold:court:1:99")
    update = _make_update(query)

    await supreme_court_callback(update, _make_context())

    query.edit_message_text.assert_awaited_once_with("Court session expired.")


# ── T4: unknown action shows "Unknown court action." ─────────────────────────

@pytest.mark.asyncio
async def test_supreme_court_callback_unknown_action():
    key = "court:1:1"
    approval_store.create_approval(key, "supreme_court", ["accept", "uphold", "overturn"])

    query = _make_callback_query(f"court_unknown:{key}")
    update = _make_update(query)

    await supreme_court_callback(update, _make_context())

    query.edit_message_text.assert_awaited_once_with("Unknown court action.")


# ── T5: strategy_callback parses "strategy_approve:<key>" correctly ──────────

@pytest.mark.asyncio
async def test_strategy_callback_approve():
    """T5: strategy_approve action sets decision to 'approve'."""
    key = "123:42"
    approval_store.create_approval(key, "strategy", ["approve", "nosplit", "cancel"])

    query = _make_callback_query(f"strategy_approve:{key}")
    query.message = MagicMock()
    query.message.text = "Strategy Report"
    update = _make_update(query)

    await strategy_callback(update, _make_context())

    approval = approval_store.get_approval(key)
    assert approval is not None
    assert approval.decision == "approve"
    assert approval.status == "decided"


# ── T6: strategy_callback with nosplit decision ───────────────────────────────

@pytest.mark.asyncio
async def test_strategy_callback_nosplit():
    """T6: strategy_nosplit action sets decision to 'nosplit'."""
    key = "123:42"
    approval_store.create_approval(key, "strategy", ["approve", "nosplit", "cancel"])

    query = _make_callback_query(f"strategy_nosplit:{key}")
    query.message = MagicMock()
    query.message.text = "Strategy Report"
    update = _make_update(query)

    await strategy_callback(update, _make_context())

    approval = approval_store.get_approval(key)
    assert approval is not None
    assert approval.decision == "nosplit"
    assert approval.status == "decided"


# ── T7: strategy_callback on already-decided shows status message ─────────────

@pytest.mark.asyncio
async def test_strategy_callback_already_decided_shows_message():
    """T7: Second click shows 'Decision already made' message."""
    key = "123:42"
    approval_store.create_approval(key, "strategy", ["approve", "nosplit", "cancel"])
    approval_store.respond(key, "approve")

    query = _make_callback_query(f"strategy_nosplit:{key}")
    query.message = MagicMock()
    query.message.text = "Strategy Report"
    update = _make_update(query)

    await strategy_callback(update, _make_context())

    query.edit_message_text.assert_awaited_once()
    call_text = query.edit_message_text.call_args[1].get("text") or query.edit_message_text.call_args[0][0]
    assert "Decision already made" in call_text or "APPROVE" in call_text


# ── T8: strategy_callback on missing key returns silently ────────────────────

@pytest.mark.asyncio
async def test_strategy_callback_missing_key_returns_silently():
    """T8: Non-existent approval key — handler returns without raising."""
    query = _make_callback_query("strategy_approve:nonexistent:99")
    query.message = MagicMock()
    query.message.text = "Strategy Report"
    update = _make_update(query)

    # Should not raise; edit_message_text shows unavailable message
    await strategy_callback(update, _make_context())
    # No assertion on edit_message_text — behavior: message is updated with suffix
