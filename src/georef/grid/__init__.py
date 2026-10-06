"""Detección de la cuadrícula del mapa."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .frame import frame_from_contour, frame_from_lines
from .lines import GridParams, detect_lines

if TYPE_CHECKING:
    from ..session import MapSession

MODES = ("lines", "ticks", "crosses")


def detect_grid(session: "MapSession", mode: str = "lines", **overrides: Any) -> dict[str, Any]:
    """Detecta la cuadrícula y actualiza la sesión. Invalida lo que dependía de la anterior."""
    mode = (mode or "lines").lower()
    if mode not in MODES:
        raise ValueError(f"Modo desconocido: {mode}. Opciones: {', '.join(MODES)}")
    params = GridParams(**{k: v for k, v in overrides.items() if v is not None})

    if mode == "lines":
        lines, info = detect_lines(session.gray, params)
    else:
        from .ticks import detect_crosses, detect_ticks

        detector = detect_ticks if mode == "ticks" else detect_crosses
        lines, info = detector(session, params)

    session.lines = lines
    session.grid_info = info
    if mode == "lines" or session.frame is None:
        session.frame = frame_from_lines(lines, session.width, session.height) or frame_from_contour(session.gray)
    session.axis_fits = {}
    session.gcps = [g for g in session.gcps if g.source == "agent"]
    session.fit = None
    for lab in session.labels:
        lab.line_id, lab.status = None, "unassigned"
    return info
