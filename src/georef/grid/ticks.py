"""Cuadrículas sin líneas completas: cruces en las intersecciones o marcas sobre el marco.

En ambos casos se construyen líneas virtuales que pasan por los cruces o unen
las marcas de lados opuestos, para que el resto del flujo no cambie.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.signal import find_peaks

from ..session import GridLine
from .lines import GridParams, _make_line, binarize, detect_lines, number_lines, ridge_centre, shrink

if TYPE_CHECKING:
    from ..session import MapSession


# --- Cruces ---


def _run(mask: np.ndarray, x: int, y: int, dx: int, dy: int, limit: int) -> int:
    """Largo del trazo desde (x, y) en la dirección (dx, dy)."""
    h, w = mask.shape
    n = 0
    while n < limit:
        x, y = x + dx, y + dy
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            break
        n += 1
    return n


def _refine_cross(gray: np.ndarray, x: float, y: float, arm: float, min_contrast: float) -> tuple[float, float] | None:
    """Centro sub-píxel de una cruz: cada brazo se mide lejos del cruce con el otro."""
    h, w = gray.shape
    window = max(5, int(round(0.5 * arm)))
    inner, outer = max(2, int(round(0.3 * arm))), max(4, int(round(0.85 * arm)))
    for _ in range(2):
        cx, cy = int(round(x)), int(round(y))
        if not (outer + window < cx < w - outer - window and outer + window < cy < h - outer - window):
            return None
        rows = slice(cy - window, cy + window + 1)
        cols = slice(cx - window, cx + window + 1)
        # Brazo horizontal: perfil vertical promediado sobre sus dos mitades.
        left, right = gray[rows, cx - outer : cx - inner], gray[rows, cx + inner : cx + outer]
        py = np.concatenate([left, right], axis=1).astype(np.float32).mean(axis=1)
        up, down = gray[cy - outer : cy - inner, cols], gray[cy + inner : cy + outer, cols]
        px = np.concatenate([up, down], axis=0).astype(np.float32).mean(axis=0)
        ry = ridge_centre(py, window, window, min_contrast)
        rx = ridge_centre(px, window, window, min_contrast)
        if rx is None or ry is None:
            return None
        x, y = cx - window + rx, cy - window + ry
    return x, y


def _points_skew(pts: np.ndarray, max_deg: float) -> float:
    """Ángulo que alinea los cruces en filas y columnas."""

    def score(angle: float, tol: float) -> int:
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        u = pts[:, 0] * c + pts[:, 1] * s
        v = -pts[:, 0] * s + pts[:, 1] * c
        return int((np.abs(u[:, None] - u[None, :]) < tol).sum() + (np.abs(v[:, None] - v[None, :]) < tol).sum())

    # Primero grueso y tolerante; luego fino y exigente, que es lo que fija el ángulo.
    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.1)
    best = float(coarse[int(np.argmax([score(a, 4.0) for a in coarse]))])
    fine = np.arange(best - 0.15, best + 0.15 + 1e-9, 0.01)
    return float(fine[int(np.argmax([score(a, 1.5) for a in fine]))])


def _trimmed_line(t: np.ndarray, u: np.ndarray) -> tuple[float, float, np.ndarray]:
    """u = m·t + q descartando los puntos que no son de esta fila o columna."""
    keep = np.ones(len(t), bool)
    m, q = 0.0, float(np.median(u))
    for _ in range(4):
        if keep.sum() >= 2 and np.ptp(t[keep]) > 1:
            m, q = np.polyfit(t[keep], u[keep], 1)
        res = u - (m * t + q)
        mad = float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        new_keep = np.abs(res) <= max(1.0, 2.5 * 1.4826 * mad)
        if new_keep.sum() < 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return float(m), float(q), keep


def _clusters(values: np.ndarray, tol: float) -> list[np.ndarray]:
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for i in order[1:]:
        if values[i] - values[current[-1]] <= tol:
            current.append(i)
        else:
            groups.append(np.array(current))
            current = [i]
    groups.append(np.array(current))
    return groups


def detect_crosses(session: "MapSession", params: GridParams) -> tuple[list[GridLine], dict]:
    gray = session.gray
    small, scale = shrink(gray, params.work_side)
    bw = cv2.dilate(binarize(small), np.ones((3, 3), np.uint8))
    k = max(7, int(0.004 * max(small.shape)))
    hmap = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))) > 0
    vmap = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))) > 0
    n, _, _, centroids = cv2.connectedComponentsWithStats((hmap & vmap).astype(np.uint8))

    limit = 6 * k
    candidates = []
    for cx, cy in centroids[1:]:
        x, y = int(round(cx)), int(round(cy))
        arms = [_run(hmap, x, y, -1, 0, limit), _run(hmap, x, y, 1, 0, limit),
                _run(vmap, x, y, 0, -1, limit), _run(vmap, x, y, 0, 1, limit)]
        # Una cruz tiene cuatro brazos cortos y parejos; una letra o un cruce de líneas no.
        if max(arms) >= limit or min(arms) < 0.35 * k or min(arms) < 0.7 * max(arms):
            continue
        candidates.append((cx, cy, float(np.mean(arms))))

    # Un mapa puede traer dos familias de cruces (retícula geográfica y cuadrícula métrica)
    # que se distinguen por su tamaño. Se agrupan por tamaño y se trabaja con una sola.
    classes: list[list[tuple[float, float, float]]] = []
    for cand in sorted(candidates, key=lambda c: c[2]):
        if classes and cand[2] <= 1.25 * classes[-1][0][2] + 1.0:
            classes[-1].append(cand)
        else:
            classes.append([cand])
    classes = sorted((c for c in classes if len(c) >= 4), key=len, reverse=True)
    summary = [{"class": i, "arm_px": round(float(np.median([c[2] for c in cls])) / scale, 1), "count": len(cls)}
               for i, cls in enumerate(classes)]
    points = []
    if classes:
        chosen = classes[min(max(0, params.cross_class), len(classes) - 1)]
        for cx, cy, arm in chosen:
            refined = _refine_cross(gray, (cx + 0.5) / scale - 0.5, (cy + 0.5) / scale - 0.5,
                                    arm / scale, params.min_contrast)
            if refined:
                points.append(refined)

    info = {"mode": "crosses", "work_scale": round(scale, 4), "candidates": int(n - 1), "crosses": len(points),
            "size_classes": summary, "cross_class": params.cross_class}
    if len(points) < 4:
        return [], {**info, "vertical": 0, "horizontal": 0}

    pts = np.array(points)
    angle = _points_skew(pts, params.max_skew_deg)
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    u = pts[:, 0] * c + pts[:, 1] * s
    v = -pts[:, 0] * s + pts[:, 1] * c
    tol = max(3.0, 0.0015 * max(gray.shape))

    columns = [g for g in _clusters(u, tol) if len(g) >= 2]
    rows = [g for g in _clusters(v, tol) if len(g) >= 2]
    # Una columna real reúne casi tantas cruces como filas hay; dos puntos sueltos alineados no.
    sizes = sorted(len(g) for g in columns + rows)
    minimum = max(2, int(0.4 * sizes[len(sizes) // 2])) if sizes else 2
    columns = [g for g in columns if len(g) >= minimum]
    rows = [g for g in rows if len(g) >= minimum]
    most = max((len(g) for g in columns + rows), default=1)

    lines: list[GridLine] = []
    for group in columns:  # x = m·y + q
        m, q, keep = _trimmed_line(pts[group, 1], pts[group, 0])
        ys = pts[group, 1][keep]
        lines.append(_make_line("v", m, q, float(ys.min()), float(ys.max()), int(keep.sum()) / most, "cross"))
    for group in rows:  # y = m·x + q
        m, q, keep = _trimmed_line(pts[group, 0], pts[group, 1])
        xs = pts[group, 0][keep]
        lines.append(_make_line("h", m, q, float(xs.min()), float(xs.max()), int(keep.sum()) / most, "cross"))
    lines = number_lines(lines, session.width, session.height)
    info.update(skew_deg=round(angle, 3), vertical=len(columns), horizontal=len(rows))
    return lines, info


# --- Marcas sobre el marco ---


def _inner_border(lines: list[GridLine], size: int, w: int, h: int) -> tuple[GridLine, GridLine] | None:
    """Las dos líneas del marco de una familia; con borde doble, la interior de cada lado."""
    if len(lines) < 2:
        return None
    ordered = sorted(lines, key=lambda ln: ln.pos(w, h))
    first, last = ordered[0].pos(w, h), ordered[-1].pos(w, h)
    near = [ln for ln in ordered if ln.pos(w, h) - first <= 0.05 * size]
    far = [ln for ln in ordered if last - ln.pos(w, h) <= 0.05 * size]
    return (near[-1], far[0]) if near[-1] is not far[0] else None


def _side_ticks(gray: np.ndarray, side: GridLine, start: float, end: float, band: int, min_contrast: float) -> list[float]:
    """Posiciones (a lo largo del lado) de las marcas perpendiculares a un lado horizontal del marco."""
    h, w = gray.shape
    xs = np.arange(int(np.ceil(start)), int(np.floor(end)))
    if len(xs) < 10:
        return []
    ys = np.array([side.y_at(float(x)) for x in xs])
    profiles = []
    for offsets in (range(4, band + 1), range(-band, -3)):
        rows = np.clip(np.round(ys[None, :] + np.array(list(offsets))[:, None]).astype(int), 0, h - 1)
        profiles.append(gray[rows, xs[None, :]].astype(np.float32).mean(axis=0))
    darkness = np.max([np.percentile(p, 70) - p for p in profiles], axis=0)
    peaks, _ = find_peaks(darkness, height=min_contrast, distance=4, prominence=min_contrast)

    found = []
    for p in peaks:
        lo, hi = max(0, p - 6), min(len(xs), p + 7)
        best = profiles[int(np.argmax([np.percentile(q, 70) - q[p] for q in profiles]))]
        centre = ridge_centre(best[lo:hi], p - lo, 6, min_contrast)
        if centre is not None:
            found.append(float(xs[lo] + centre))
    return found


def _transpose(line: GridLine) -> GridLine:
    """La misma recta con los ejes intercambiados, para reutilizar el código de un lado horizontal."""
    return GridLine(line.id, "h" if line.axis == "v" else "v", line.b, line.a, line.c,
                    (line.p0[1], line.p0[0]), (line.p1[1], line.p1[0]), line.support, line.kind)


def detect_ticks(session: "MapSession", params: GridParams) -> tuple[list[GridLine], dict]:
    gray = session.gray
    w, h = session.width, session.height
    border, _ = detect_lines(gray, params)
    vertical = _inner_border([ln for ln in border if ln.axis == "v"], w, w, h)
    horizontal = _inner_border([ln for ln in border if ln.axis == "h"], h, w, h)
    info = {"mode": "ticks", "vertical": 0, "horizontal": 0}
    if not vertical or not horizontal:
        return [], {**info, "note": "No se encontró el marco del mapa."}

    left, right = vertical
    top, bottom = horizontal
    band = max(10, int(0.006 * max(w, h)))
    tol = max(4.0, 0.004 * max(w, h))
    lines: list[GridLine] = []

    # Marcas arriba y abajo -> líneas verticales.
    x0, x1 = left.pos(w, h) + band, right.pos(w, h) - band
    upper = _side_ticks(gray, top, x0, x1, band, params.min_contrast)
    lower = _side_ticks(gray, bottom, x0, x1, band, params.min_contrast)
    drift = (left.p1[0] - left.p0[0]) / ((left.p1[1] - left.p0[1]) or 1.0)  # dx/dy del marco
    for xt in upper:
        yt = top.y_at(xt)
        expected = [xb - drift * (bottom.y_at(xb) - yt) for xb in lower]
        if not expected:
            break
        j = int(np.argmin(np.abs(np.array(expected) - xt)))
        if abs(expected[j] - xt) <= tol:
            xb, yb = lower[j], bottom.y_at(lower[j])
            m = (xb - xt) / (yb - yt)
            lines.append(_make_line("v", m, xt - m * yt, yt, yb, 1.0, "tick"))

    # Marcas a izquierda y derecha -> líneas horizontales (mismo cálculo con los ejes intercambiados).
    gray_t = np.ascontiguousarray(gray.T)
    left_t, right_t = _transpose(left), _transpose(right)
    y0, y1 = top.pos(w, h) + band, bottom.pos(w, h) - band
    west = _side_ticks(gray_t, left_t, y0, y1, band, params.min_contrast)
    east = _side_ticks(gray_t, right_t, y0, y1, band, params.min_contrast)
    drift_h = (top.p1[1] - top.p0[1]) / ((top.p1[0] - top.p0[0]) or 1.0)  # dy/dx del marco
    for yl in west:
        xl = left.x_at(yl)
        expected = [yr - drift_h * (right.x_at(yr) - xl) for yr in east]
        if not expected:
            break
        j = int(np.argmin(np.abs(np.array(expected) - yl)))
        if abs(expected[j] - yl) <= tol:
            yr, xr = east[j], right.x_at(east[j])
            m = (yr - yl) / (xr - xl)
            lines.append(_make_line("h", m, yl - m * xl, xl, xr, 1.0, "tick"))

    lines = number_lines(lines, w, h)
    info.update(vertical=sum(ln.axis == "v" for ln in lines), horizontal=sum(ln.axis == "h" for ln in lines),
                ticks={"top": len(upper), "bottom": len(lower), "left": len(west), "right": len(east)})
    return lines, info
