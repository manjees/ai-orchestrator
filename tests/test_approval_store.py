"""Unit tests for orchestrator/approval_store.py — shared approval state."""
from __future__ import annotations

import asyncio

import pytest

from orchestrator import approval_store
from orchestrator.approval_store import ApprovalStatus, ApprovalType, make_approval_id


@pytest.fixture(autouse=True)
def cleanup_store():
    yield
    approval_store.clear_all()


# ── T1: create_approval returns Approval with correct fields ──────────────────

def test_create_approval_returns_approval_with_fields():
    approval = approval_store.create_approval(
        "key1", "strategy", ["approve", "nosplit", "cancel"]
    )
    assert approval.approval_id == "key1"
    assert approval.type == "strategy"
    assert approval.status == "pending"
    assert approval.decision is None
    assert approval.decision_options == ["approve", "nosplit", "cancel"]
    assert isinstance(approval.event, asyncio.Event)


# ── T2: create_approval with explicit key uses that key ───────────────────────

def test_create_approval_uses_provided_key():
    key = "custom_key_123"
    approval = approval_store.create_approval(key, "strategy", ["approve"])
    assert approval.approval_id == key


# ── T3: get_approval returns existing approval ────────────────────────────────

def test_get_approval_existing():
    approval_store.create_approval("key1", "strategy", ["approve"])
    result = approval_store.get_approval("key1")
    assert result is not None
    assert result.approval_id == "key1"


# ── T4: get_approval returns None for missing key ─────────────────────────────

def test_get_approval_missing_returns_none():
    assert approval_store.get_approval("nonexistent") is None


# ── T5: respond sets decision and fires event ─────────────────────────────────

@pytest.mark.asyncio
async def test_respond_sets_decision_and_fires_event():
    approval = approval_store.create_approval(
        "key1", "strategy", ["approve", "cancel"]
    )
    result = approval_store.respond("key1", "approve")
    assert result is True
    assert approval.decision == "approve"
    assert approval.status == "decided"
    assert approval.event.is_set()


# ── T6: respond on already-decided returns False ──────────────────────────────

@pytest.mark.asyncio
async def test_respond_already_decided_returns_false():
    approval = approval_store.create_approval("key1", "strategy", ["approve", "cancel"])
    approval_store.respond("key1", "approve")
    result = approval_store.respond("key1", "cancel")
    assert result is False
    assert approval.decision == "approve"


# ── T7: respond on non-existent approval returns False ───────────────────────

@pytest.mark.asyncio
async def test_respond_nonexistent_returns_false():
    result = approval_store.respond("nonexistent", "approve")
    assert result is False


# ── T8: respond with invalid decision raises ValueError ──────────────────────

@pytest.mark.asyncio
async def test_respond_invalid_decision_raises():
    approval_store.create_approval("key1", "strategy", ["approve", "cancel"])
    with pytest.raises(ValueError, match="Invalid decision"):
        approval_store.respond("key1", "invalid_choice")


# ── T9: remove_approval cleans up store ──────────────────────────────────────

def test_remove_approval_cleanup():
    approval_store.create_approval("key1", "strategy", ["approve"])
    approval_store.remove_approval("key1")
    assert approval_store.get_approval("key1") is None


# ── T10: list_pending returns only pending approvals ─────────────────────────

def test_list_pending_filters_decided():
    approval_store.create_approval("pending_key", "strategy", ["approve"])
    approval_store.create_approval("decided_key", "strategy", ["approve"])
    approval_store.respond("decided_key", "approve")
    pending = approval_store.list_pending()
    ids = [a.approval_id for a in pending]
    assert "pending_key" in ids
    assert "decided_key" not in ids


# ── T11: wait_for_decision returns decision after respond ─────────────────────

@pytest.mark.asyncio
async def test_wait_for_decision_returns_after_respond():
    approval_store.create_approval("key1", "strategy", ["approve"])

    async def respond_later():
        await asyncio.sleep(0.05)
        approval_store.respond("key1", "approve")

    asyncio.create_task(respond_later())
    decision = await approval_store.wait_for_decision("key1", timeout=2.0)
    assert decision == "approve"


# ── T12: wait_for_decision returns None on timeout ────────────────────────────

@pytest.mark.asyncio
async def test_wait_for_decision_timeout_returns_none():
    approval_store.create_approval("key1", "strategy", ["approve"])
    decision = await approval_store.wait_for_decision("key1", timeout=0.05)
    assert decision is None


# ── T13: context is stored correctly ─────────────────────────────────────────

def test_create_approval_stores_context():
    ctx = {"issue_num": 42, "triage_result": {"model": "haiku"}}
    approval = approval_store.create_approval("key1", "strategy", ["approve"], context=ctx)
    assert approval.context == ctx


# ── T14: wait_for_decision on non-existent key returns None ──────────────────

@pytest.mark.asyncio
async def test_wait_for_decision_nonexistent_returns_none():
    result = await approval_store.wait_for_decision("nonexistent", timeout=0.05)
    assert result is None


# ── T15: list_pending excludes expired approvals ─────────────────────────────

@pytest.mark.asyncio
async def test_list_pending_excludes_expired():
    approval_store.create_approval("expiring_key", "strategy", ["approve"])
    await approval_store.wait_for_decision("expiring_key", timeout=0.01)
    pending = approval_store.list_pending()
    ids = [a.approval_id for a in pending]
    assert "expiring_key" not in ids


# ── T16: ApprovalType enum has expected members ───────────────────────────────

def test_approval_type_enum_members():
    assert ApprovalType.STRATEGY == "strategy"
    assert ApprovalType.SUPREME_COURT == "supreme_court"


# ── T17: ApprovalStatus enum has expected members ────────────────────────────

def test_approval_status_enum_members():
    assert ApprovalStatus.PENDING == "pending"
    assert ApprovalStatus.DECIDED == "decided"
    assert ApprovalStatus.EXPIRED == "expired"


# ── T18: make_approval_id produces expected format ───────────────────────────

def test_make_approval_id_format():
    result = make_approval_id(chat_id=123, issue_num=42)
    assert result == "123:42"


# ── T19: make_approval_id with prefix ────────────────────────────────────────

def test_make_approval_id_with_prefix():
    result = make_approval_id(chat_id=123, issue_num=42, prefix="court")
    assert result == "court:123:42"


# ── T20: create_approval with ApprovalType enum ──────────────────────────────

def test_create_approval_accepts_enum_type():
    approval = approval_store.create_approval(
        "key1", ApprovalType.SUPREME_COURT, ["uphold", "overturn"]
    )
    assert approval.type == ApprovalType.SUPREME_COURT


# ── T21: Approval status transitions use enum ────────────────────────────────

def test_respond_sets_decided_status_enum():
    approval_store.create_approval("key1", ApprovalType.STRATEGY, ["approve"])
    approval_store.respond("key1", "approve")
    approval = approval_store.get_approval("key1")
    assert approval.status == ApprovalStatus.DECIDED


# ── T22: wait_for_decision timeout sets expired enum ─────────────────────────

@pytest.mark.asyncio
async def test_wait_timeout_sets_expired_status_enum():
    approval_store.create_approval("key1", ApprovalType.STRATEGY, ["approve"])
    await approval_store.wait_for_decision("key1", timeout=0.01)
    approval = approval_store.get_approval("key1")
    assert approval.status == ApprovalStatus.EXPIRED


# ── T23: list_pending works with supreme_court type ──────────────────────────

def test_list_pending_includes_supreme_court_type():
    approval_store.create_approval(
        "court:1:5", ApprovalType.SUPREME_COURT,
        ["accept", "uphold", "overturn"],
        context={"ruling": "UPHOLD"},
    )
    pending = approval_store.list_pending()
    assert len(pending) == 1
    assert pending[0].type == ApprovalType.SUPREME_COURT
