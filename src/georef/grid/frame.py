"""Marco del mapa (el rectángulo donde termina la cuadrícula)."""

from __future__ import annotations

import cv2
import numpy as np

from ..session import Frame, GridLine
from .lines import binarize, shrink


def frame_from_lines(lines: list[GridLine], width: int, height: int) -> Frame | None:
    """El marco se deduce de dónde terminan las líneas de la cuadrícula."""
    vertical = [ln for ln in lines if ln.axis == "v" and ln.on_lattice]
    horizontal = [ln for ln in lines if ln.axis == "h" and ln.on_lattice]
    if not vertical and not horizontal:
        return None

    if horizontal:
        x0 = float(np.median([min(ln.p0[0], ln.p1[0]) for ln in horizontal]))
        x1 = float(np.median([max(ln.p0[0], ln.p1[0]) for ln in horizontal]))
    else:
        xs = [ln.pos(width, height) for ln in vertical]
        x0, x1 = min(xs), max(xs)
    if vertical:
        y0 = float(np.median([min(ln.p0[1], ln.p1[1]) for ln in vertical]))
        y1 = float(np.median([max(ln.p0[1], ln.p1[1]) for ln in vertical]))
    else:
        ys = [ln.pos(width, height) for ln in horizontal]
        y0, y1 = min(ys), max(ys)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return Frame(x0, y0, x1, y1, source="grid")


def frame_from_contour(gray: np.ndarray) -> Frame | None:
    """Sin cuadrícula: el mayor contorno cerrado de la hoja."""
    small, scale = shrink(gray, 1600)
    bw = cv2.morphologyEx(binarize(small), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    if w * h < 0.2 * small.shape[0] * small.shape[1]:
        return None
    return Frame(x / scale, y / scale, (x + w) / scale, (y + h) / scale, source="contour")
