"""Dibujo sobre las vistas: reglas de píxeles, rejilla de referencia y grilla detectada."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..session import MapSession

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT = cv2.FONT_HERSHEY_SIMPLEX
RULER_TOP = 24
RULER_LEFT = 56

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
COLOR_V = (255, 90, 0)  # azul
COLOR_H = (0, 0, 230)  # rojo
COLOR_OFF = (150, 150, 150)
COLOR_GCP = (0, 170, 0)
COLOR_RESIDUAL = (0, 140, 255)
LABEL_COLORS = {"ok": (0, 170, 0), "outlier": (0, 0, 255), "unassigned": (0, 200, 230)}


@dataclass
class RefGrid:
    """Rejilla fija sobre la imagen original. Las celdas se nombran A1, B1, ..."""

    cell: int
    cols: int
    rows: int
    width: int
    height: int

    @classmethod
    def for_image(cls, width: int, height: int, target: int = 10) -> "RefGrid":
        cell = max(1, math.ceil(max(width, height) / target))
        return cls(cell, math.ceil(width / cell), math.ceil(height / cell), width, height)

    def name(self, col: int, row: int) -> str:
        return f"{LETTERS[col]}{row + 1}"

    def cell_at(self, x: float, y: float) -> str:
        col = int(min(self.cols - 1, max(0, x // self.cell)))
        row = int(min(self.rows - 1, max(0, y // self.cell)))
        return self.name(col, row)

    def _index(self, token: str) -> tuple[int, int]:
        token = token.strip().upper()
        if len(token) < 2 or token[0] not in LETTERS or not token[1:].isdigit():
            raise ValueError(f'Celda inválida: "{token}". Formato esperado: C4 o B2:C3.')
        col, row = LETTERS.index(token[0]), int(token[1:]) - 1
        if col >= self.cols or not 0 <= row < self.rows:
            last = self.name(self.cols - 1, self.rows - 1)
            raise ValueError(f'La celda "{token}" está fuera de la rejilla (A1 a {last}).')
        return col, row

    def parse(self, text: str) -> tuple[int, int, int, int]:
        """Celda ("C4") o rango ("B2:C3") a (x, y, w, h) en píxeles."""
        first, _, last = text.partition(":")
        c0, r0 = self._index(first)
        c1, r1 = self._index(last) if last else (c0, r0)
        c0, c1 = sorted((c0, c1))
        r0, r1 = sorted((r0, r1))
        x0, y0 = c0 * self.cell, r0 * self.cell
        x1 = min(self.width, (c1 + 1) * self.cell)
        y1 = min(self.height, (r1 + 1) * self.cell)
        return x0, y0, x1 - x0, y1 - y0

    def describe(self) -> str:
        last = self.name(self.cols - 1, self.rows - 1)
        return f"{self.cols} columnas x {self.rows} filas (A1 a {last}), celdas de {self.cell} px"


def _nice_step(target: float) -> int:
    """Menor paso "redondo" (1, 2, 5 x 10^k) que no baja de `target`."""
    target = max(1.0, target)
    exp = 10 ** math.floor(math.log10(target))
    for m in (1, 2, 5, 10):
        if m * exp >= target:
            return int(m * exp)
    return int(10 * exp)


def _text(img: np.ndarray, text: str, org: tuple[int, int], color=BLACK, scale=0.4, bg=None) -> None:
    if bg is not None:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw_ref_grid(view: np.ndarray, x0: float, y0: float, scale: float, grid: RefGrid) -> None:
    """Dibuja sobre `view` las celdas de la rejilla que caen dentro de ella."""
    h, w = view.shape[:2]
    layer = view.copy()
    x_end, y_end = x0 + w / scale, y0 + h / scale
    for col in range(grid.cols + 1):
        gx = min(col * grid.cell, grid.width)
        if x0 <= gx <= x_end:
            vx = int(round((gx - x0) * scale))
            cv2.line(layer, (vx, 0), (vx, h - 1), MAGENTA, 1)
    for row in range(grid.rows + 1):
        gy = min(row * grid.cell, grid.height)
        if y0 <= gy <= y_end:
            vy = int(round((gy - y0) * scale))
            cv2.line(layer, (0, vy), (w - 1, vy), MAGENTA, 1)
    cv2.addWeighted(layer, 0.55, view, 0.45, 0, dst=view)

    for col in range(grid.cols):
        for row in range(grid.rows):
            cx0, cy0 = col * grid.cell, row * grid.cell
            if cx0 + grid.cell <= x0 or cx0 >= x_end or cy0 + grid.cell <= y0 or cy0 >= y_end:
                continue
            vx = int(round((max(cx0, x0) - x0) * scale)) + 4
            vy = int(round((max(cy0, y0) - y0) * scale)) + 15
            _text(view, grid.name(col, row), (vx, vy), MAGENTA, 0.45, bg=WHITE)


def add_rulers(view: np.ndarray, x0: float, y0: float, scale: float) -> np.ndarray:
    """Añade un borde con reglas en píxeles de la imagen original."""
    h, w = view.shape[:2]
    canvas = np.full((h + RULER_TOP, w + RULER_LEFT, 3), 255, np.uint8)
    canvas[RULER_TOP:, RULER_LEFT:] = view
    step = _nice_step(110 / scale)

    gx = math.ceil(x0 / step) * step
    while gx < x0 + w / scale:
        vx = RULER_LEFT + int(round((gx - x0) * scale))
        cv2.line(canvas, (vx, RULER_TOP - 7), (vx, RULER_TOP - 1), BLACK, 1)
        _text(canvas, str(gx), (vx + 3, RULER_TOP - 9), scale=0.38)
        gx += step

    gy = math.ceil(y0 / step) * step
    while gy < y0 + h / scale:
        vy = RULER_TOP + int(round((gy - y0) * scale))
        cv2.line(canvas, (RULER_LEFT - 7, vy), (RULER_LEFT - 1, vy), BLACK, 1)
        _text(canvas, str(gy), (2, vy - 3 if vy > RULER_TOP + 12 else vy + 12), scale=0.38)
        gy += step
    return canvas


def draw_grid_overlay(
    view: np.ndarray, x0: float, y0: float, scale: float, session: "MapSession", show: set[str]
) -> None:
    """Grilla detectada con el ID y el valor de cada línea, etiquetas, puntos y residuos."""
    from ..coords.parse import format_value

    h, w = view.shape[:2]

    def to_view(x: float, y: float) -> tuple[int, int]:
        return int(round((x - x0) * scale)), int(round((y - y0) * scale))

    if "labels" in show:
        for lab in session.labels:
            p0, p1 = to_view(lab.x0, lab.y0), to_view(lab.x1, lab.y1)
            if p1[0] < 0 or p1[1] < 0 or p0[0] > w or p0[1] > h:
                continue
            color = LABEL_COLORS.get(lab.status, LABEL_COLORS["unassigned"])
            cv2.rectangle(view, p0, p1, color, 1)
            _text(view, lab.id, (p0[0], max(10, p0[1] - 3)), color, 0.36, bg=WHITE)

    if "lines" in show:
        for ln in session.lines:
            color = COLOR_OFF if not ln.on_lattice else (COLOR_V if ln.axis == "v" else COLOR_H)
            a, b = to_view(*ln.p0), to_view(*ln.p1)
            ok, a, b = cv2.clipLine((0, 0, w, h), a, b)
            if not ok:
                continue
            cv2.line(view, a, b, color, 2 if ln.value is not None else 1, cv2.LINE_AA)
            tag = ln.id
            if "values" in show and ln.value is not None:
                tag += "=" + format_value(session.label_kind or "utm", ln.value)
            first = a if (a[1] <= b[1] if ln.axis == "v" else a[0] <= b[0]) else b
            org = (first[0] + 4, first[1] + 14) if ln.axis == "v" else (first[0] + 4, first[1] - 4)
            org = (min(max(2, org[0]), w - 60), min(max(12, org[1]), h - 4))
            _text(view, tag, org, color, 0.4, bg=WHITE)

    if "gcps" in show or "residuals" in show:
        residuals = getattr(session.fit, "residual_vectors", {}) if session.fit else {}
        for g in session.gcps:
            p = to_view(g.px, g.py)
            if not (0 <= p[0] < w and 0 <= p[1] < h):
                continue
            cv2.circle(view, p, 4, COLOR_GCP if g.enabled else COLOR_OFF, 1, cv2.LINE_AA)
            if "residuals" in show and g.id in residuals:
                dx, dy = residuals[g.id]
                # Los residuos son de fracciones de píxel: se amplían para que se vean.
                tip = (int(round(p[0] + dx * 25)), int(round(p[1] + dy * 25)))
                cv2.line(view, p, tip, COLOR_RESIDUAL, 2, cv2.LINE_AA)


def encode_image(image: np.ndarray, quality: int = 85) -> tuple[str, str]:
    """Devuelve (media_type, base64). JPEG para vistas en color, PNG si es binaria."""
    is_binary = image.size > 0 and len(np.unique(image[::7, ::7])) <= 2
    ext, params = (".png", []) if is_binary else (".jpg", [cv2.IMWRITE_JPEG_QUALITY, quality])
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen.")
    media = "image/png" if is_binary else "image/jpeg"
    return media, base64.standard_b64encode(buf.tobytes()).decode("ascii")


def image_block(image: np.ndarray, quality: int = 85) -> dict[str, Any]:
    media, data = encode_image(image, quality)
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def render_view(
    session: "MapSession",
    region: tuple[int, int, int, int],
    max_side: int,
    enhance: str = "none",
    ref_grid: bool = True,
    overlay: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Vista de una región con reglas. Devuelve la imagen y sus metadatos."""
    from .enhance import apply_enhance

    x, y, w, h = region
    content_side = max(256, max_side - RULER_LEFT)
    view, scale = session.pyramid.view(x, y, w, h, content_side)
    view = apply_enhance(view, enhance)
    if overlay:
        draw_grid_overlay(view, x, y, scale, session, overlay)
    if ref_grid:
        draw_ref_grid(view, x, y, scale, session.ref_grid)
    canvas = add_rulers(view, x, y, scale)
    meta = {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "scale": round(scale, 4),
        "view_size": [canvas.shape[1], canvas.shape[0]],
        "note": "Los números de los bordes son píxeles de la imagen original.",
    }
    return canvas, meta
