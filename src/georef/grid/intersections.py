"""Intersecciones entre líneas de la cuadrícula."""

from __future__ import annotations

from ..session import GridLine


def intersect(v: GridLine, h: GridLine) -> tuple[float, float] | None:
    """Punto de cruce de dos rectas, o None si son casi paralelas."""
    det = v.a * h.b - h.a * v.b
    if abs(det) < 1e-9:
        return None
    x = (v.b * h.c - h.b * v.c) / det
    y = (h.a * v.c - v.a * h.c) / det
    return x, y
