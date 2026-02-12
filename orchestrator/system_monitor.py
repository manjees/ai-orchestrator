"""System monitoring via psutil and macOS pmset."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from functools import partial

import psutil

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemStatus:
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    cpu_percent: float
    thermal_pressure: str
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float


def _get_thermal_pressure() -> str:
    """Parse CPU_Speed_Limit from ``pmset -g therm`` (no sudo required)."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "CPU_Speed_Limit" in line:
                value = int(line.split("=")[-1].strip())
                if value >= 100:
                    return "nominal"
                if value >= 75:
                    return "moderate"
                if value >= 50:
                    return "heavy"
                return "critical"
    except Exception:
        logger.debug("Failed to read thermal pressure", exc_info=True)
    return "unavailable"


def _collect_status() -> SystemStatus:
    """Blocking helper — run in executor."""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return SystemStatus(
        ram_total_gb=round(vm.total / (1024**3), 1),
        ram_used_gb=round(vm.used / (1024**3), 1),
        ram_percent=vm.percent,
        cpu_percent=psutil.cpu_percent(interval=0.5),
        thermal_pressure=_get_thermal_pressure(),
        disk_total_gb=round(disk.total / (1024**3), 1),
        disk_used_gb=round(disk.used / (1024**3), 1),
        disk_percent=disk.percent,
    )


async def get_system_status() -> SystemStatus:
    """Async wrapper that offloads blocking calls to a thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _collect_status)
