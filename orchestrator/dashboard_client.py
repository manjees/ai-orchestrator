"""Fire-and-forget async HTTP client for orchestra-dashboard REST API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

AGENT_REGISTRY: list[dict[str, str]] = [
    {"agent_id": "orchestrator",      "type": "ORCHESTRATOR", "name": "AI Orchestrator"},
    {"agent_id": "haiku-researcher",  "type": "WORKER",       "name": "Haiku (Research)"},
    {"agent_id": "opus-designer",     "type": "PLANNER",      "name": "Opus (Design)"},
    {"agent_id": "gemini-reviewer",   "type": "REVIEWER",     "name": "Gemini (Critique/Court)"},
    {"agent_id": "sonnet-worker",     "type": "WORKER",       "name": "Sonnet (Implement/Review)"},
    {"agent_id": "deepseek-auditor",  "type": "REVIEWER",     "name": "DeepSeek R1 (Audit)"},
    {"agent_id": "qwen-worker",       "type": "WORKER",       "name": "Qwen (Hints/Mining)"},
]


class DashboardClient:
    """Async client for dashboard event reporting.

    All public methods are fire-and-forget: exceptions are caught and logged,
    never propagated. Dashboard unavailability must never interrupt the pipeline.
    """

    def __init__(self, base_url: str, timeout: int = 5, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

    async def register_agents(self) -> None:
        """POST /api/v1/agents for each pipeline agent."""
        for agent in AGENT_REGISTRY:
            try:
                resp = await self._client.post("/api/v1/agents", json=agent)
                if resp.status_code >= 400:
                    logger.warning(
                        "Failed to register agent %s: %d %s",
                        agent["agent_id"], resp.status_code, resp.text,
                    )
            except httpx.HTTPError:
                logger.warning(
                    "Dashboard unreachable for agent registration: %s",
                    agent["agent_id"],
                )
            except Exception:
                logger.warning(
                    "Unexpected error registering agent: %s",
                    agent["agent_id"], exc_info=True,
                )

    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """PATCH /api/v1/agents/{id}/status."""
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        if not status:
            raise ValueError("status must not be empty")
        try:
            resp = await self._client.patch(
                f"/api/v1/agents/{agent_id}/status",
                json={"status": status},
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Agent status update failed: %s -> %s (%d)",
                    agent_id, status, resp.status_code,
                )
        except httpx.HTTPError:
            logger.warning("Dashboard unreachable for status update: %s", agent_id)
        except Exception:
            logger.warning("Unexpected error updating agent status: %s", agent_id, exc_info=True)

    async def send_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """POST /api/v1/events."""
        if not event_type:
            raise ValueError("event_type must not be empty")
        payload = {"event_type": event_type, "data": data or {}}
        try:
            resp = await self._client.post("/api/v1/events", json=payload)
            if resp.status_code >= 400:
                logger.warning("Event send failed: %s (%d)", event_type, resp.status_code)
        except httpx.HTTPError:
            logger.warning("Dashboard unreachable for event: %s", event_type)
        except Exception:
            logger.warning("Unexpected error sending event: %s", event_type, exc_info=True)

    async def create_pipeline_run(self, data: dict[str, Any]) -> str | None:
        """POST /api/v1/pipeline-runs, returns run ID or None on failure."""
        try:
            resp = await self._client.post("/api/v1/pipeline-runs", json=data)
            if resp.status_code < 400:
                return resp.json().get("id")
            logger.warning("Pipeline run creation failed: %d", resp.status_code)
        except httpx.HTTPError:
            logger.warning("Dashboard unreachable for pipeline run creation")
        except Exception:
            logger.warning("Unexpected error creating pipeline run", exc_info=True)
        return None

    async def update_pipeline_run(self, run_id: str, data: dict[str, Any]) -> None:
        """PATCH /api/v1/pipeline-runs/{id}."""
        if not run_id:
            raise ValueError("run_id must not be empty")
        try:
            resp = await self._client.patch(
                f"/api/v1/pipeline-runs/{run_id}", json=data,
            )
            if resp.status_code >= 400:
                logger.warning("Pipeline run update failed: %s (%d)", run_id, resp.status_code)
        except httpx.HTTPError:
            logger.warning("Dashboard unreachable for pipeline run update: %s", run_id)
        except Exception:
            logger.warning("Unexpected error updating pipeline run: %s", run_id, exc_info=True)

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        await self._client.aclose()
