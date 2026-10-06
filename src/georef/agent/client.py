"""Cliente del modelo."""

from __future__ import annotations

import os
from typing import Any

from .. import tracing
from ..config import Settings


def make_client(settings: Settings) -> Any:
    """Claude en Vertex AI con las credenciales de GCP. `GEOREF_PROVIDER=anthropic` usa la API directa."""
    if os.getenv("GEOREF_PROVIDER", "vertex").strip().lower() == "anthropic":
        import anthropic

        client: Any = anthropic.Anthropic()
    else:
        from anthropic import AnthropicVertex

        if not settings.gcp_project:
            raise ValueError("Falta GCP_PROJECT en el entorno.")
        client = AnthropicVertex(project_id=settings.gcp_project, region=settings.vertex_region)
    return tracing.wrap_client(client)
