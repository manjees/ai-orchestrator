"""Pydantic response models for the API."""

from pydantic import BaseModel


class OllamaModelInfo(BaseModel):
    name: str
    size_gb: float


class TmuxSessionInfo(BaseModel):
    name: str
    windows: int
    created: str


class StatusResponse(BaseModel):
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    cpu_percent: float
    thermal_pressure: str
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    ollama_models: list[OllamaModelInfo]
    tmux_sessions: list[TmuxSessionInfo]
