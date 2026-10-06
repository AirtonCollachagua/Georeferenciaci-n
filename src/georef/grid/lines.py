"""Detección de las líneas de la cuadrícula con precisión sub-píxel.

Dos etapas. La gruesa trabaja a resolución reducida: corrige la inclinación del
escaneo, aísla trazos largos con morfología y los localiza por perfiles de
proyección. La fina vuelve a la resolución original y ajusta cada recta sobre
el centro de la tinta, medido en decenas de estaciones a lo largo de la línea.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths

from ..session import GridLine


@dataclass
class GridParams:
    work_side: int = 3000  # lado largo de la etapa gruesa
    kernel_frac: float = 0.06  # largo mínimo de un trazo continuo, relativo al lado menor
    gap_frac: float = 0.0  # cortes que se puentean; subirlo para cuadrículas de trazos
    max_thickness: int = 8  # grosor máximo del trazo en píxeles de trabajo; el texto es más grueso
    min_span: float = 0.30  # extensión mínima de la línea respecto al lado de la imagen
    min_support: float = 0.35  # fracción mínima de la línea cubierta por tinta
    peak_rel: float = 0.35  # altura mínima de un pico frente a las líneas más largas
    max_skew_deg: float = 6.0
    min_contrast: float = 18.0  # niveles de gris entre la línea y el fondo
    cross_class: int = 0  # modo crosses: 0 = el tamaño de cruz más frecuente, 1 = el segundo, ...


def binarize(gray: np.ndarray) -> np.ndarray:
    """Tinta en blanco sobre fondo negro."""
    block = max(15, int(0.012 * max(gray.shape)) | 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12
    )


def shrink(gray: np.ndarray, work_side: int) -> tuple[np.ndarray, float]:
    """Reduce la imagen conservando los trazos finos y oscuros."""
    scale = min(1.0, work_side / max(gray.shape))
    if scale >= 1.0:
        return gray, 1.0
    k = int(round(1 / scale))
    if k > 1:
        gray = cv2.erode(gray, np.ones((k, k), np.uint8))
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def estimate_skew(bw: np.ndarray, max_deg: float) -> float:
    """Ángulo (grados) que hay que girar la imagen para dejar la grilla alineada a los ejes."""
    f = min(1.0, 1200 / max(bw.shape))
    small = cv2.resize(bw, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else bw
    small = cv2.dilate((small > 60).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    h, w = small.shape
    length = max(11, int(0.02 * min(h, w)))
    horiz = cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vert = cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    if not horiz.any() and not vert.any():
        return 0.0

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        ph = cv2.warpAffine(horiz, m, (w, h)).sum(axis=1, dtype=np.float64)
        pv = cv2.warpAffine(vert, m, (w, h)).sum(axis=0, dtype=np.float64)
        return float((ph**2).sum() + (pv**2).sum())

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.25)
    best = coarse[int(np.argmax([score(a) for a in coarse]))]
    fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.02)
    return float(fine[int(np.argmax([score(a) for a in fine]))])


def _rotation(shape: tuple[int, int], angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Matriz de giro que no recorta las esquinas, y el tamaño del lienzo resultante."""
    h, w = shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(w * cos + h * sin)), int(np.ceil(w * sin + h * cos))
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    return m, (nw, nh)


def _coarse_family(mask: np.ndarray, kernel: int, params: GridParams) -> list[tuple[float, float, float]]:
    """Líneas verticales de `mask`: lista de (x, y_inicio, y_fin) en píxeles de trabajo."""
    h, w = mask.shape
    gap = int(params.gap_frac * min(h, w))
    bridged = mask
    if gap >= 2:
        bridged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
    long_strokes = cv2.morphologyEx(
        bridged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel))
    )
    ink = long_strokes > 0
    profile = np.convolve(ink.sum(axis=0).astype(float), np.ones(3), mode="same")
    distance = max(3, int(0.004 * w))
    peaks, props = find_peaks(profile, height=kernel, distance=distance)
    if len(peaks) == 0:
        return []
    reference = float(np.median(np.sort(props["peak_heights"])[-5:]))
    # Una fila de texto también deja un pico, pero mucho más ancho que una línea.
    widths = peak_widths(profile, peaks, rel_height=0.5)[0]
    found = []
    for x, height, width in zip(peaks, props["peak_heights"], widths):
        if height < params.peak_rel * reference or width > params.max_thickness:
            continue
        rows = np.flatnonzero(ink[:, max(0, x - 1) : x + 2].any(axis=1))
        if len(rows) == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        length = y1 - y0 + 1
        if length < params.min_span * h or len(rows) / length < params.min_support:
            continue
        found.append((float(x), float(y0), float(y1)))
    return found


def ridge_centre(profile: np.ndarray, expected: int, window: int, min_contrast: float) -> float | None:
    """Centro sub-píxel del trazo oscuro en un perfil de grises perpendicular a la línea."""
    depth = float(np.percentile(profile, 85)) - profile
    # Ante dos valles parecidos gana el más cercano a donde se espera la línea.
    centred = depth * (1 - 0.3 * np.abs(np.arange(len(depth)) - expected) / max(1, window))
    k = int(np.argmax(centred))
    peak = float(depth[k])
    if peak < min_contrast:
        return None
    lo = hi = k
    while lo > 0 and depth[lo - 1] > 0.4 * peak:
        lo -= 1
    while hi < len(depth) - 1 and depth[hi + 1] > 0.4 * peak:
        hi += 1
    if hi - lo + 1 > max(6, window):  # mancha ancha, no una línea
        return None
    idx = np.arange(lo, hi + 1)
    weight = depth[idx] - 0.4 * peak
    if weight.sum() <= 0:
        return None
    return float((idx * weight).sum() / weight.sum())


def _refine_vertical(
    gray: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    half_window: int,
    min_contrast: float,
    stations: int = 64,
) -> tuple[float, float, float, float] | None:
    """Ajusta x = m·y + q sobre el centro de la tinta. Devuelve (m, q, soporte, rms)."""
    h, w = gray.shape
    (x0, y0), (x1, y1) = sorted((p0, p1), key=lambda p: p[1])
    if y1 - y0 < 8:
        return None
    m = (x1 - x0) / (y1 - y0)
    q = x0 - m * y0
    margin = 0.02 * (y1 - y0)
    ys = np.linspace(y0 + margin, y1 - margin, stations)

    result = None
    for window in (half_window, 5):
        pts = []
        for y in ys:
            cy, cx = int(round(y)), int(round(m * y + q))
            ya, yb = max(0, cy - 3), min(h, cy + 4)
            xa, xb = max(0, cx - window), min(w, cx + window + 1)
            if xb - xa < 5 or yb <= ya:
                continue
            profile = gray[ya:yb, xa:xb].astype(np.float32).mean(axis=0)
            centre = ridge_centre(profile, cx - xa, window, min_contrast)
            if centre is not None:
                pts.append((y, xa + centre))
        if len(pts) < max(6, int(0.15 * stations)):
            return None
        arr = np.array(pts)
        keep = np.ones(len(arr), bool)
        for _ in range(4):
            m, q = np.polyfit(arr[keep, 0], arr[keep, 1], 1)
            res = arr[:, 1] - (m * arr[:, 0] + q)
            mad = float(np.median(np.abs(res[keep] - np.median(res[keep]))))
            new_keep = np.abs(res) <= max(0.75, 2.5 * 1.4826 * mad)
            if new_keep.sum() < 6 or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        rms = float(np.sqrt(np.mean((arr[keep, 1] - (m * arr[keep, 0] + q)) ** 2)))
        result = (float(m), float(q), float(keep.sum() / stations), rms)
    return result


def _make_line(axis: str, m: float, q: float, t0: float, t1: float, support: float, kind: str) -> GridLine:
    """Construye la línea a partir de u = m·t + q (vertical: u=x, t=y; horizontal: u=y, t=x)."""
    norm = float(np.hypot(1.0, m))
    if axis == "v":
        a, b, c = 1 / norm, -m / norm, -q / norm
        p0, p1 = (m * t0 + q, t0), (m * t1 + q, t1)
    else:
        a, b, c = -m / norm, 1 / norm, -q / norm
        p0, p1 = (t0, m * t0 + q), (t1, m * t1 + q)
    return GridLine("", axis, a, b, c, p0, p1, support=support, kind=kind)


def flag_lattice(positions: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Marca las líneas que caen en una retícula de paso constante.

    Las vías, los bordes y el marco rara vez respetan el paso de la cuadrícula.
    """
    n = len(positions)
    if n < 3:
        return np.ones(n, bool), None
    best_mask, best_count, best_step = np.ones(n, bool), 0, None
    for step in sorted(set(np.round(np.diff(positions), 1)), reverse=True):
        if step < 4:
            continue
        for anchor in positions:
            k = np.round((positions - anchor) / step)
            err = np.abs(positions - anchor - k * step)
            # Tolerancia ajustada, con una holgura que crece con la distancia al ancla
            # para admitir la deformación del papel.
            mask = err < max(2.5, 0.012 * step) + 0.003 * step * np.abs(k)
            if mask.sum() > best_count:  # a igual número de aciertos gana el paso mayor
                best_mask, best_count, best_step = mask, int(mask.sum()), float(step)
    if best_count < 3 or best_count < 0.5 * n:
        return np.ones(n, bool), None
    return best_mask, best_step


def number_lines(lines: list[GridLine], width: int, height: int) -> list[GridLine]:
    """Ordena, numera (V1.., H1..) y marca la pertenencia a la retícula."""
    out: list[GridLine] = []
    for axis, prefix in (("v", "V"), ("h", "H")):
        family = sorted((ln for ln in lines if ln.axis == axis), key=lambda ln: ln.pos(width, height))
        mask, _ = flag_lattice(np.array([ln.pos(width, height) for ln in family]))
        for i, (ln, on) in enumerate(zip(family, mask), start=1):
            ln.id = f"{prefix}{i}"
            ln.on_lattice = bool(on)
            out.append(ln)
    return out


def detect_lines(gray: np.ndarray, params: GridParams | None = None) -> tuple[list[GridLine], dict]:
    """Líneas de la cuadrícula en píxeles de la imagen original."""
    params = params or GridParams()
    height, width = gray.shape
    small, scale = shrink(gray, params.work_side)
    bw = binarize(small)
    skew = estimate_skew(bw, params.max_skew_deg)

    rot, (nw, nh) = _rotation(bw.shape, skew)
    bw_r = cv2.warpAffine(bw, rot, (nw, nh))
    bw_r = cv2.dilate((bw_r > 127).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    inv = cv2.invertAffineTransform(rot)
    kernel = max(25, int(params.kernel_frac * min(bw.shape)))

    def to_full(x: float, y: float) -> tuple[float, float]:
        xs, ys = inv @ np.array([x, y, 1.0])
        return (xs + 0.5) / scale - 0.5, (ys + 0.5) / scale - 0.5

    half_window = max(6, int(np.ceil(4 / scale)))
    lines: list[GridLine] = []
    coarse_count = {"v": 0, "h": 0}

    for x, y0, y1 in _coarse_family(bw_r, kernel, params):
        coarse_count["v"] += 1
        p0, p1 = to_full(x, y0), to_full(x, y1)
        fit = _refine_vertical(gray, p0, p1, half_window, params.min_contrast)
        if fit and fit[2] >= params.min_support:
            m, q, support, _ = fit
            lines.append(_make_line("v", m, q, min(p0[1], p1[1]), max(p0[1], p1[1]), support, "line"))

    # La familia horizontal se resuelve igual, con los ejes intercambiados.
    for y, x0, x1 in _coarse_family(np.ascontiguousarray(bw_r.T), kernel, params):
        coarse_count["h"] += 1
        p0, p1 = to_full(x0, y), to_full(x1, y)
        fit = _refine_vertical(gray.T, (p0[1], p0[0]), (p1[1], p1[0]), half_window, params.min_contrast)
        if fit and fit[2] >= params.min_support:
            m, q, support, _ = fit
            lines.append(_make_line("h", m, q, min(p0[0], p1[0]), max(p0[0], p1[0]), support, "line"))

    lines = number_lines(lines, width, height)
    info = {
        "mode": "lines",
        "skew_deg": round(skew, 3),
        "work_scale": round(scale, 4),
        "coarse": coarse_count,
        "vertical": sum(ln.axis == "v" for ln in lines),
        "horizontal": sum(ln.axis == "h" for ln in lines),
    }
    return lines, info
