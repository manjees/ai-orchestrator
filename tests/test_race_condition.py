"""Race-condition tests: Telegram + API competing for the same approval."""
from __future__ import annotations

import asyncio

import pytest

from orchestrator import approval_store


@pytest.fixture(autouse=True)
def cleanup_store():
    yield
    approval_store.clear_all()


# ── T1: Two concurrent respond() calls — only the first wins ─────────────────

@pytest.mark.asyncio
async def test_concurrent_respond_only_first_wins():
    """T1: Two simultaneous respond() — exactly one returns True."""
    approval_store.create_approval("key1", "strategy", ["approve", "cancel"])

    results = await asyncio.gather(
        asyncio.to_thread(approval_store.respond, "key1", "approve"),
        asyncio.to_thread(approval_store.respond, "key1", "cancel"),
    )
    # Exactly one must succeed; the other sees DECIDED and returns False
    assert sorted(results) == [False, True]
    approval = approval_store.get_approval("key1")
    assert approval is not None
    assert approval.status == "decided"


# ── T2: API POST + Telegram respond() race — first-writer-wins ───────────────

@pytest.mark.asyncio
async def test_api_and_telegram_race_first_writer_wins():
    """T2: Simulated Telegram + API concurrent respond — only one succeeds."""
    approval_store.create_approval("key1", "strategy", ["approve", "cancel"])

    result_api = approval_store.respond("key1", "approve")
    result_telegram = approval_store.respond("key1", "cancel")

    assert result_api is True
    assert result_telegram is False
    approval = approval_store.get_approval("key1")
    assert approval is not None
    assert approval.decision == "approve"


# ── T3: Concurrent respond() on already-expired approval ─────────────────────

@pytest.mark.asyncio
async def test_concurrent_respond_on_expired_returns_false():
    """T3: Expired approval — all concurrent respond() calls return False."""
    approval_store.create_approval("key1", "strategy", ["approve", "cancel"])
    await approval_store.wait_for_decision("key1", timeout=0.01)

    results = await asyncio.gather(
        asyncio.to_thread(approval_store.respond, "key1", "approve"),
        asyncio.to_thread(approval_store.respond, "key1", "cancel"),
    )
    assert results == [False, False]


# ── T4: Rapid sequential respond() — second is rejected ─────────────────────

@pytest.mark.asyncio
async def test_rapid_sequential_respond_second_rejected():
    """T4: Second respond() call returns False, first decision is preserved."""
    approval_store.create_approval("key1", "strategy", ["approve", "cancel"])

    r1 = approval_store.respond("key1", "approve")
    r2 = approval_store.respond("key1", "cancel")

    assert r1 is True
    assert r2 is False
    assert approval_store.get_approval("key1").decision == "approve"  # type: ignore[union-attr]


# ── T5: wait_for_decision unblocks regardless of source ──────────────────────

@pytest.mark.asyncio
async def test_wait_unblocks_from_any_respond_source():
    """T5: wait_for_decision returns when respond() called (simulating API path)."""
    approval_store.create_approval("key1", "strategy", ["approve"])

    async def api_side_respond():
        await asyncio.sleep(0.05)
        approval_store.respond("key1", "approve")

    asyncio.create_task(api_side_respond())
    decision = await approval_store.wait_for_decision("key1", timeout=2.0)
    assert decision == "approve"
