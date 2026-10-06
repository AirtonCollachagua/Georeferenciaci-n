"""Asociación etiqueta <-> línea y reconstrucción de la grilla completa con sus valores.

En una cuadrícula el valor de cada línea es una función lineal de su posición.
Con dos o más etiquetas bien leídas se ajusta esa relación, se descartan las
lecturas que no encajan y se deduce el valor de las líneas sin etiqueta.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..grid.intersections import intersect
from ..session import GCP, AxisFit, CoordLabel, GridLine
from .parse import parse_candidates

if TYPE_CHECKING:
    from ..session import MapSession

FAMILY_OF_AXIS = {"x": "v", "y": "h"}
_GEO_STEPS = [s / 3600 for s in (1, 2, 5, 10, 15, 20, 30, 60, 120, 150, 300, 600, 900, 1200, 1800,
                                  3600, 7200, 18000, 36000)]


def build_labels(session: "MapSession") -> list[CoordLabel]:
    """Convierte el índice OCR en etiquetas de coordenada y las ubica respecto al marco."""
    labels: list[CoordLabel] = []
    for index, item in enumerate(session.ocr.items, start=1):
        candidates = parse_candidates(item.text)
        if len(candidates) != 1:
            continue  # texto corrido o varias coordenadas juntas: lo resuelve el agente
        c = candidates[0]
        labels.append(
            CoordLabel(
                # El ID sale de la posición en el índice OCR: no cambia al leer más texto.
                id=f"L{index}", text=item.text, kind=c.kind, value=c.value, axis=c.axis,
                complete=c.complete, x0=item.x0, y0=item.y0, x1=item.x1, y1=item.y1,
                conf=item.conf, digits=c.digits,
            )
        )
    for lab in labels:
        lab.side = _side(lab, session)
    session.labels = labels
    return labels


def _side(lab: CoordLabel, session: "MapSession") -> str:
    frame = session.frame
    if frame is None:
        return ""
    band_x = 0.04 * (frame.x1 - frame.x0)
    band_y = 0.04 * (frame.y1 - frame.y0)
    out_left, out_right = lab.cx < frame.x0, lab.cx > frame.x1
    out_top, out_bottom = lab.cy < frame.y0, lab.cy > frame.y1
    if (out_left or out_right) and (out_top or out_bottom):
        return "corner"
    if out_top:
        return "top"
    if out_bottom:
        return "bottom"
    if out_left:
        return "left"
    if out_right:
        return "right"
    # Dentro del marco: cuenta como margen si está pegada a un borde.
    distances = {
        "top": (lab.cy - frame.y0) / band_y, "bottom": (frame.y1 - lab.cy) / band_y,
        "left": (lab.cx - frame.x0) / band_x, "right": (frame.x1 - lab.cx) / band_x,
    }
    side, d = min(distances.items(), key=lambda kv: kv[1])
    return side if d <= 1.0 else "inside"


def _spacing(lines: list[GridLine], session: "MapSession") -> float | None:
    pos = sorted(ln.pos(session.width, session.height) for ln in lines)
    if len(pos) < 2:
        return None
    return float(np.median(np.diff(pos)))


def _candidate_lines(session: "MapSession", family: str) -> list[GridLine]:
    lines = session.lines_of(family)
    on_lattice = [ln for ln in lines if ln.on_lattice]
    return on_lattice or lines


def associate(session: "MapSession") -> None:
    """Asigna a cada etiqueta la línea cuya prolongación pasa más cerca."""
    for lab in session.labels:
        lab.line_id, lab.status = None, "unassigned"

    for family in ("v", "h"):
        lines = _candidate_lines(session, family)
        if not lines:
            continue
        spacing = _spacing(lines, session)
        size = session.width if family == "v" else session.height
        tol = 0.3 * spacing if spacing else 0.05 * size
        for lab in session.labels:
            if lab.side in ("top", "bottom"):
                wanted = "v"
            elif lab.side in ("left", "right"):
                wanted = "h"
            elif lab.side == "inside" and lab.axis:
                wanted = FAMILY_OF_AXIS[lab.axis]
            else:
                continue
            if wanted != family or (lab.axis and FAMILY_OF_AXIS[lab.axis] != family):
                continue
            if family == "v":
                dist = [abs(ln.x_at(lab.cy) - lab.cx) for ln in lines]
            else:
                dist = [abs(ln.y_at(lab.cx) - lab.cy) for ln in lines]
            k = int(np.argmin(dist))
            if dist[k] <= tol:
                lab.line_id = lines[k].id


def _nice_step(raw: float, kind: str) -> tuple[float, bool]:
    """Paso "redondo" más cercano y si de verdad queda cerca."""
    if raw <= 0:
        return raw, False
    if kind == "geo":
        nice = min(_GEO_STEPS, key=lambda s: abs(math.log(s / raw)))
    else:
        exp = 10 ** math.floor(math.log10(raw))
        nice = min((m * exp for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(math.log(s / raw)))
    return nice, abs(nice - raw) / nice < 0.08


def _robust_line(pos: np.ndarray, val: np.ndarray, tol_per_px: float) -> tuple[float, float, np.ndarray] | None:
    """valor = pendiente·posición + ordenada, tolerando lecturas erróneas."""
    best = None
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(pos[j] - pos[i]) < 1e-6:
                continue
            slope = (val[j] - val[i]) / (pos[j] - pos[i])
            if slope == 0:
                continue
            intercept = val[i] - slope * pos[i]
            err = np.abs(slope * pos + intercept - val)
            inliers = err < abs(slope) * tol_per_px
            key = (int(inliers.sum()), -float(err[inliers].sum()))
            if best is None or key > best[0]:
                best = (key, inliers)
    if best is None:
        return None
    inliers = best[1]
    if len(set(np.round(pos[inliers], 1))) < 2:
        return None
    slope, intercept = np.polyfit(pos[inliers], val[inliers], 1)
    return float(slope), float(intercept), inliers


def _step_from_values(session, axis, kind, slope, labelled, inliers, anchors, spacing) -> float:
    """Paso de la grilla: la menor diferencia entre valores de líneas rotuladas distintas."""
    by_line: dict[str, float] = {}
    for lab, ok in zip(labelled, inliers):
        if ok:
            by_line[lab.line_id] = abs(lab.value) if kind == "geo" else lab.value
    known = sorted(set(round(v, 9) for v in by_line.values()) | {round(abs(v) if kind == "geo" else v, 9) for _, v in anchors})
    diffs = [b - a for a, b in zip(known, known[1:]) if b - a > 1e-9]
    raw = min(diffs) if diffs else abs(slope) * spacing
    step, is_nice = _nice_step(raw, kind)
    if not is_nice:
        session.warnings.append(
            f"Eje {axis}: el paso estimado ({raw:.6g}) no es un valor redondo; revisar las etiquetas."
        )
        return raw
    return step


def _fit_axis(session: "MapSession", axis: str, kind: str, step: float | None = None) -> AxisFit | None:
    family = FAMILY_OF_AXIS[axis]
    lines = session.lines_of(family)
    spacing = _spacing(_candidate_lines(session, family), session)
    if not lines or not spacing:
        return None
    by_id = {ln.id: ln for ln in lines}

    def pos_of(ln: GridLine) -> float:
        return ln.pos(session.width, session.height)

    anchors = [(pos_of(ln), ln.value) for ln in lines if ln.value_source == "agent" and ln.value is not None]
    labelled = [
        lab for lab in session.labels
        if lab.kind == kind and lab.complete and lab.value is not None and lab.line_id in by_id
    ]

    sign = 1
    values = np.array([lab.value for lab in labelled], dtype=float)
    if kind == "geo" and len(labelled):
        # Las etiquetas rara vez traen hemisferio: se comparan en valor absoluto
        # y el signo se deduce más abajo del sentido en que crecen.
        values = np.abs(values)
    positions = np.array([pos_of(by_id[lab.line_id]) for lab in labelled], dtype=float)

    if len(anchors) >= 2:
        # Lo que fija el agente manda sobre el OCR.
        a_pos, a_val = np.array(anchors).T
        slope, intercept = np.polyfit(a_pos, a_val, 1)
        if kind == "geo":
            sign = -1 if np.mean(a_val) < 0 else 1
        fitted = slope * positions + intercept if len(positions) else np.array([])
        compare = values * sign if kind == "geo" else values
        inliers = np.abs(fitted - compare) < abs(slope) * 0.25 * spacing if len(positions) else np.array([], bool)
    else:
        if anchors:
            a_pos, a_val = anchors[0]
            positions = np.append(positions, a_pos)
            values = np.append(values, abs(a_val) if kind == "geo" else a_val)
        if len(positions) < 2:
            return None
        fit = _robust_line(positions, values, 0.25 * spacing)
        if fit is None:
            return None
        slope, intercept, inliers = fit
        if anchors:
            inliers = inliers[:-1]
        if kind == "geo":
            signed = [lab.value for lab in labelled if lab.value < 0]
            if signed or (anchors and anchors[0][1] < 0):
                sign = -1
            elif (axis == "x" and slope < 0) or (axis == "y" and slope > 0):
                # Con el norte arriba, la longitud crece hacia la derecha y la latitud
                # hacia arriba. Si los valores sin signo van al revés, son Oeste o Sur.
                sign = -1
            slope, intercept = slope * sign, intercept * sign

    if step is None:
        step = _step_from_values(session, axis, kind, slope, labelled, inliers, anchors, spacing)

    # Una línea de la cuadrícula cae sobre un valor redondo salvo por el error de
    # medida, que son pocos píxeles. Con más holgura, un trazo cualquiera recibiría
    # una coordenada por azar.
    step_px = step / abs(slope)
    tolerance = abs(slope) * max(2.0, 0.006 * step_px)

    for lab, ok in zip(labelled, inliers):
        lab.status = "ok" if ok else "outlier"
    inlier_lines = {lab.line_id for lab, ok in zip(labelled, inliers) if ok}

    for ln in lines:
        if ln.value_source == "agent":
            continue
        predicted = slope * pos_of(ln) + intercept
        snapped = round(predicted / step) * step
        if abs(predicted - snapped) <= tolerance:
            ln.value = float(round(snapped, 9))
            ln.value_source = "label" if ln.id in inlier_lines else "fit"
            ln.on_lattice = True
        else:
            ln.value, ln.value_source = None, ""

    return AxisFit(axis, kind, float(step), float(slope), float(intercept),
                   n_labels=len(labelled), n_inliers=int(np.sum(inliers)) + len(anchors), sign=sign)


def _clear_values(session: "MapSession", keep_agent: bool) -> None:
    for ln in session.lines:
        if keep_agent and ln.value_source == "agent":
            continue
        ln.value, ln.value_source = None, ""


def fit_grid(session: "MapSession", kind: str | None = None, steps: dict[str, float] | None = None) -> dict[str, AxisFit]:
    """Ajusta ambos ejes. Si no se indica el tipo, elige el que explica más etiquetas."""
    steps = steps or {}
    session.warnings = [w for w in session.warnings if not w.startswith("Eje ")]
    kinds = [kind] if kind else ["utm", "geo"]
    best: tuple[int, str] | None = None
    def attempt(k: str) -> dict[str, AxisFit | None]:
        _clear_values(session, keep_agent=True)
        for lab in session.labels:
            lab.status = "unassigned"
        session.warnings = [w for w in session.warnings if not w.startswith("Eje ")]
        return {a: _fit_axis(session, a, k, steps.get(a)) for a in ("x", "y")}

    for k in kinds:
        fits = attempt(k)
        key = (int(all(fits.values())), sum(f.n_inliers for f in fits.values() if f))
        if best is None or key > best[0]:
            best = (key, k)
    chosen = best[1] if best else (kind or "utm")
    fits = attempt(chosen)
    session.axis_fits = {a: f for a, f in fits.items() if f}
    session.label_kind = chosen if session.axis_fits else session.label_kind
    build_gcps(session)
    return session.axis_fits


def build_gcps(session: "MapSession") -> list[GCP]:
    """Un punto de control en cada cruce de dos líneas con valor."""
    disabled = {g.id for g in session.gcps if not g.enabled}
    manual = [g for g in session.gcps if g.source == "agent"]
    frame = session.frame
    margin = 0.02 * max(session.width, session.height)
    gcps: list[GCP] = []
    for v in session.lines_of("v"):
        if v.value is None:
            continue
        for h in session.lines_of("h"):
            if h.value is None:
                continue
            pt = intersect(v, h)
            if pt is None:
                continue
            x, y = pt
            if not (0 <= x < session.width and 0 <= y < session.height):
                continue
            if frame is not None and not frame.contains(x, y, margin):
                continue
            gid = f"{v.id}{h.id}"
            gcps.append(GCP(gid, x, y, v.value, h.value, "grid", enabled=gid not in disabled))
    session.gcps = gcps + manual
    session.fit = None
    return session.gcps


def drop_inconsistent_lines(session: "MapSession", max_rounds: int = 4) -> list[str]:
    """Quita el valor a las líneas cuyos puntos se desvían todos hacia el mismo lado.

    Es la firma de una línea que no es de la cuadrícula (el marco, una vía) y que
    recibió una coordenada redonda por estar cerca de una.
    """
    from ..georef.transform import fit_transform

    dropped: list[str] = []
    for _ in range(max_rounds):
        fit = session.fit
        if fit is None or len(fit.residual_vectors) < 8:
            break
        vectors = fit.residual_vectors
        worst: tuple[float, GridLine] | None = None
        for ln in session.lines:
            if ln.value is None or ln.value_source == "agent":
                continue
            comp = 0 if ln.axis == "v" else 1
            on_line = (lambda gid: gid.startswith(ln.id + "H")) if ln.axis == "v" else (lambda gid: gid.endswith(ln.id))
            mine = np.array([v[comp] for gid, v in vectors.items() if on_line(gid)])
            rest = np.array([v[comp] for gid, v in vectors.items() if not on_line(gid)])
            if len(mine) < 2 or len(rest) < 4:
                continue
            spread = 1.4826 * float(np.median(np.abs(rest - np.median(rest))))
            offset = float(np.median(mine))
            same_side = bool((np.sign(mine) == np.sign(offset)).all())
            if same_side and abs(offset) > max(1.5, 4 * spread) and (worst is None or abs(offset) > worst[0]):
                worst = (abs(offset), ln)
        if worst is None:
            break
        ln = worst[1]
        ln.value, ln.value_source = None, ""
        dropped.append(ln.id)
        build_gcps(session)
        if sum(g.enabled for g in session.gcps) < 3:
            break
        fit_transform(session, "auto")
    if dropped:
        session.warnings.append("Líneas descartadas por no encajar en la cuadrícula: " + ", ".join(dropped))
    return dropped
