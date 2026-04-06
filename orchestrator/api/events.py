"""WebSocket EventBus — singleton that broadcasts events to connected clients."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Event Types ───────────────────────────────────────────────────────────────


class EventType:
    PIPELINE_STARTED = "pipeline.started"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_RESPONDED = "approval.responded"
    SYSTEM_STATUS = "system.status"


class WSEvent(BaseModel):
    event_type: str
    data: dict[str, Any]
    timestamp: str  # ISO 8601


# ── EventBus Singleton ───────────────────────────────────────────────────────


class EventBus:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._status_task: asyncio.Task | None = None

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        event = WSEvent(
            event_type=event_type,
            data=data or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        payload = event.model_dump_json()
        dead: list[WebSocket] = []

        async with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

        # JSONL logging (fire-and-forget, sync write in executor)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, get_event_logger().append, event_type, data or {},
            )
        except Exception:
            logger.warning("JSONL append failed", exc_info=True)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start_status_loop(self, interval: float = 10.0) -> None:
        """Start periodic system.status emission."""
        self._status_task = asyncio.create_task(self._status_loop(interval))

    async def stop_status_loop(self) -> None:
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass

    async def _status_loop(self, interval: float) -> None:
        while True:
            try:
                status = await get_system_status()
                await self.emit(EventType.SYSTEM_STATUS, {
                    "ram_percent": status.ram_percent,
                    "cpu_percent": status.cpu_percent,
                    "active_pipelines": len(registry_list_all()),
                })
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("system.status emit failed", exc_info=True)
            await asyncio.sleep(interval)


# Lazy imports to avoid circular dependencies
def get_event_logger():
    from orchestrator.event_logger import get_event_logger as _get
    return _get()


def get_system_status():
    from orchestrator.system_monitor import get_system_status as _get
    return _get()


def registry_list_all() -> dict:
    from orchestrator.api import registry
    return registry.list_all()


# ── Module-level singleton ───────────────────────────────────────────────────

_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """For testing only."""
    global _event_bus
    _event_bus = None
