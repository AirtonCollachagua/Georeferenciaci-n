"""Configuración leída del entorno (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "on", "yes", "si", "sí"}


def _list(name: str) -> list[str]:
    return [p.strip() for p in os.getenv(name, "").split(",") if p.strip()]


@dataclass
class Settings:
    # Google Cloud
    gcp_project: str = ""
    vertex_region: str = "global"
    docai_location: str = "us"
    docai_processor_id: str = ""

    # Agente
    model: str = "claude-opus-5-5"
    effort: str = "high"
    max_turns: int = 40
    max_tokens: int = 16000

    # Trazas
    tracing: bool = False
    trace_images: bool = True

    # Pistas de CRS
    region_hint: list[str] = field(default_factory=list)
    default_datum: str = ""

    # Vistas enviadas al modelo (lado largo en píxeles)
    view_side: int = 1568
    view_side_max: int = 2576

    # Umbrales para dar un mapa por bueno sin revisión
    ok_rmse_px: float = 2.0
    ok_min_gcps: int = 6

    cache_dir: Path = Path(".cache")

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Settings":
        if env_file and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        return cls(
            gcp_project=os.getenv("GCP_PROJECT", ""),
            vertex_region=os.getenv("VERTEX_REGION", "global"),
            docai_location=os.getenv("DOCAI_LOCATION", "us"),
            docai_processor_id=os.getenv("DOCAI_PROCESSOR_ID", ""),
            model=os.getenv("CLAUDE_MODEL", "claude-opus-5-5"),
            effort=os.getenv("CLAUDE_EFFORT", "high"),
            max_turns=int(os.getenv("GEOREF_MAX_TURNS", "40")),
            tracing=_flag("LANGSMITH_TRACING") and bool(os.getenv("LANGSMITH_API_KEY")),
            trace_images=_flag("GEOREF_TRACE_IMAGES", True),
            region_hint=_list("GEOREF_REGION_HINT"),
            default_datum=os.getenv("GEOREF_DEFAULT_DATUM", "").strip().upper(),
            cache_dir=Path(os.getenv("GEOREF_CACHE_DIR", ".cache")),
        )
