"""Bucle del agente sobre un mapa."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import anthropic

from .. import tracing
from ..config import Settings
from ..imaging.annotate import image_block, render_view
from ..session import MapSession
from .prompts import SYSTEM_PROMPT, initial_text
from .tools import build_tools

_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class TraceWriter:
    """Copia local de lo que hizo el agente, sin imágenes."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def write(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.time(), 3), "type": kind, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _log_message(trace: TraceWriter, turn: int, message: Any) -> None:
    for block in message.content:
        if block.type == "thinking" and getattr(block, "thinking", ""):
            trace.write("thinking", turn=turn, text=block.thinking)
        elif block.type == "text" and block.text.strip():
            trace.write("text", turn=turn, text=block.text)
        elif block.type == "tool_use":
            trace.write("tool_use", turn=turn, name=block.name, input=block.input)


def _log_results(trace: TraceWriter, turn: int, response: Any) -> None:
    for block in (response or {}).get("content", []):
        if block.get("type") != "tool_result":
            continue
        content = block.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        images = sum(p.get("type") == "image" for p in parts)
        trace.write("tool_result", turn=turn, is_error=bool(block.get("is_error")), text=text[:4000], images=images)


@tracing.traced("agent")
def run_agent(session: MapSession, settings: Settings, client: Any, prepass: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Deja que el modelo compruebe y corrija el pre-análisis. Devuelve el informe de la corrida."""
    trace = TraceWriter(Path(out_dir) / "trace.jsonl")
    overview, _ = render_view(session, (0, 0, session.width, session.height), settings.view_side, ref_grid=True)
    messages = [{
        "role": "user",
        "content": [image_block(overview), {"type": "text", "text": initial_text(session.path.name, prepass)}],
    }]
    report: dict[str, Any] = {"used": True, "model": settings.model, "effort": settings.effort,
                              "turns": 0, "stop_reason": None, "submitted": False,
                              "usage": dict.fromkeys(_USAGE_FIELDS, 0)}
    try:
        runner = client.beta.messages.tool_runner(
            model=settings.model,
            max_tokens=settings.max_tokens,
            max_iterations=settings.max_turns,
            system=SYSTEM_PROMPT,
            tools=build_tools(session, settings),
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": settings.effort},
            cache_control={"type": "ephemeral"},
        )
        for message in runner:
            report["turns"] += 1
            report["stop_reason"] = message.stop_reason
            for name in _USAGE_FIELDS:
                report["usage"][name] += getattr(message.usage, name, 0) or 0
            _log_message(trace, report["turns"], message)
            if message.stop_reason == "refusal":
                break
            # Ejecuta las herramientas de este turno (el runner reutiliza el resultado).
            _log_results(trace, report["turns"], runner.generate_tool_call_response())
            if session.result is not None:
                break
    except anthropic.NotFoundError as exc:
        report["error"] = f"El modelo {settings.model} no está disponible para este proyecto: {exc.message}"
    except anthropic.RateLimitError as exc:
        report["error"] = f"Límite de uso alcanzado: {exc.message}"
    except anthropic.APIStatusError as exc:
        report["error"] = f"Error {exc.status_code} de la API: {exc.message}"
    except anthropic.APIConnectionError as exc:
        report["error"] = f"Sin conexión con la API: {exc}"

    report["submitted"] = session.result is not None
    if "error" in report:
        session.warnings.append("El agente no pudo ejecutarse: " + report["error"])
    elif not report["submitted"]:
        reason = "rechazo del modelo" if report["stop_reason"] == "refusal" else "límite de turnos o fin sin cierre"
        session.warnings.append(f"El agente terminó sin llamar a submit_result ({reason}).")
        if report["stop_reason"] == "refusal":
            session.result = {"status": "failed", "confidence": 0.0, "notes": "El modelo rechazó la solicitud."}
    trace.write("end", **{k: v for k, v in report.items() if k != "used"})
    return report
