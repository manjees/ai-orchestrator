"""JSONL event logger with file rotation."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_DIR = Path("data")
DEFAULT_EVENTS_FILE = "events.jsonl"
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


class EventLogger:
    def __init__(
        self,
        directory: Path = DEFAULT_EVENTS_DIR,
        filename: str = DEFAULT_EVENTS_FILE,
        max_size: int = MAX_FILE_SIZE_BYTES,
    ):
        self._dir = directory
        self._filepath = directory / filename
        self._max_size = max_size

    def append(self, event_type: str, data: dict[str, Any]) -> None:
        """Append one JSON line to the events file."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._maybe_rotate()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "pipeline_id": data.get("pipeline_id", ""),
            "project_name": data.get("project_name", ""),
            "data": data,
        }
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_rotate(self) -> None:
        if self._filepath.exists() and self._filepath.stat().st_size > self._max_size:
            rotated = self._filepath.with_suffix(".jsonl.1")
            if rotated.exists():
                rotated.unlink()
            self._filepath.rename(rotated)

    def read_pipeline_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Read recent pipeline-level events, most recent first."""
        if not self._filepath.exists():
            return []

        pipeline_event_types = {
            "pipeline.started",
            "pipeline.completed",
            "pipeline.failed",
        }
        entries: list[dict] = []
        with open(self._filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event_type") in pipeline_event_types:
                    entries.append(record)

        return list(reversed(entries[-limit:]))


# Module-level singleton
_logger: EventLogger | None = None


def get_event_logger() -> EventLogger:
    global _logger
    if _logger is None:
        _logger = EventLogger()
    return _logger


def reset_event_logger() -> None:
    """For testing."""
    global _logger
    _logger = None
