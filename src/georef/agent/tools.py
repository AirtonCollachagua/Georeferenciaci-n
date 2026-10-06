"""Herramientas del agente.

Cada herramienta opera sobre la sesión del mapa y devuelve texto JSON compacto,
más una imagen cuando hace falta verla. El estado pesado nunca viaja al modelo.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal, TypedDict

import cv2
import numpy as np
from anthropic import beta_tool

from .. import tracing
from ..config import Settings
from ..coords.crs import crs_name, propose_crs, read_legend
from ..coords.labels import associate, build_gcps, build_labels, fit_grid
from ..coords.parse import find_datum, find_scale, find_zone, format_value, parse_candidates
from ..georef.transform import fit_transform as fit_transform_impl
from ..georef.transform import rank_crs as rank_crs_impl
from ..georef.validate import validate as validate_impl
from ..grid import detect_grid as detect_grid_impl
from ..grid.frame import frame_from_contour, frame_from_lines
from ..imaging.annotate import image_block, render_view
from ..ocr.tiling import run_ocr
from ..session import GCP, MapSession

_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")
OVERLAY_LAYERS = {"lines", "values", "labels", "gcps", "residuals"}


class Assignment(TypedDict):
    line_id: str
    value: str


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_value(text: str) -> float:
    """Acepta "320000", "-76.75" o "76°30' W"."""
    t = str(text).strip()
    if _NUMERIC.match(t):
        return float(t)
    candidates = [c for c in parse_candidates(t) if c.complete and c.value is not None]
    if len(candidates) != 1:
        raise ValueError(f'No se pudo interpretar "{text}" como una coordenada.')
    return float(candidates[0].value)


def lines_brief(session: MapSession) -> dict[str, list[str]]:
    """Grilla en una línea por familia: "V3@300=319000" es la línea V3, en x=300, con valor 319000."""
    out: dict[str, list[str]] = {"V": [], "H": []}
    for ln in session.lines:
        tag = f"{ln.id}@{ln.pos(session.width, session.height):.0f}"
        if ln.value is not None:
            tag += "=" + format_value(session.label_kind or "utm", ln.value)
        elif not ln.on_lattice:
            tag += "(fuera de retícula)"
        out["V" if ln.axis == "v" else "H"].append(tag)
    return out


def fit_brief(session: MapSession) -> dict[str, Any] | None:
    if session.fit is None:
        return None
    worst = sorted((g for g in session.gcps if g.enabled and g.residual_px is not None),
                   key=lambda g: -g.residual_px)[:6]
    return {**session.fit.summary(), "worst": [{"id": g.id, "residual_px": round(g.residual_px, 2)} for g in worst]}


def snap_point(gray: np.ndarray, px: float, py: float, radius: int = 12) -> tuple[float, float]:
    """Lleva un punto aproximado al centro de la cruz, o a la esquina o cruce de trazos más cercano."""
    from ..grid.ticks import _refine_cross

    # Primero como cruz: cada brazo se mide por separado, que es lo más preciso.
    for arm in (40.0, 25.0, 60.0, 15.0):
        centre = _refine_cross(gray, px, py, arm, 18.0)
        if centre and np.hypot(centre[0] - px, centre[1] - py) <= 0.6 * arm:
            return centre
    h, w = gray.shape
    if not (radius < px < w - radius - 1 and radius < py < h - radius - 1):
        return px, py
    pt = np.array([[[px, py]]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    refined = cv2.cornerSubPix(gray, pt, (radius, radius), (-1, -1), criteria)[0, 0]
    if not np.isfinite(refined).all() or np.hypot(refined[0] - px, refined[1] - py) > radius:
        return px, py
    return float(refined[0]), float(refined[1])


def refit(session: MapSession) -> dict[str, Any] | None:
    """Reajusta si hay puntos suficientes; devuelve el resumen o None."""
    if sum(g.enabled for g in session.gcps) < 3:
        session.fit = None
        return None
    fit_transform_impl(session, "auto")
    return fit_brief(session)


def build_tools(session: MapSession, settings: Settings) -> list[Any]:
    """Herramientas ligadas a una sesión, listas para el tool runner."""

    def run(name: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
        def safe(**kw: Any) -> Any:
            try:
                return fn(**kw)
            except Exception as exc:  # el modelo recibe el motivo y puede corregir la llamada
                return _json({"error": f"{type(exc).__name__}: {exc}"})

        return tracing.run_tool(name, safe, **kwargs)

    # ------------------------------------------------------------------ vista

    def _view(region: str, max_side: int, enhance: str, ref_grid: bool, overlay: set[str] | None = None) -> list[dict]:
        rect = session.parse_region(region)
        side = int(min(settings.view_side_max, max(512, max_side)))
        canvas, meta = render_view(session, rect, side, enhance, ref_grid, overlay)
        if meta["scale"] < 0.5:
            meta["hint"] = "Vista reducida: para leer texto pequeño pide una región menor."
        return [{"type": "text", "text": _json(meta)}, image_block(canvas)]

    @beta_tool
    def view_image(region: str = "", max_side: int = 1568, enhance: str = "none", ref_grid: bool = True) -> Any:
        """Devuelve una vista del mapa: la imagen completa o un recorte ampliado (zoom).

        Los bordes de la vista llevan reglas con píxeles de la imagen original. Pedir una
        región más pequeña es hacer zoom; un recorte pequeño se amplía hasta 4 veces.

        Args:
            region: Vacío para todo el mapa; "x,y,w,h" en píxeles originales; o celdas de la rejilla de referencia ("C4" o un rango "B2:C3").
            max_side: Lado largo de la vista en píxeles (512 a 2576). Usa 1568 para ubicarte y hasta 2576 para leer texto fino.
            enhance: Realce: none, clahe, binarize, sharpen, invert, gray, red, green o blue (un canal aislado separa tintas de color).
            ref_grid: Dibuja la rejilla de referencia (celdas A1, B1, ...) para nombrar zonas.
        """
        return run("view_image", lambda **kw: _view(**kw), region=region, max_side=max_side, enhance=enhance, ref_grid=ref_grid)

    # -------------------------------------------------------------------- OCR

    def _ocr_region(region: str, rotation: int, upscale: float) -> str:
        rect = session.parse_region(region)
        items = run_ocr(session, rect, rotation=rotation, upscale=upscale)
        added = session.ocr.add(items)
        build_labels(session)
        associate(session)
        found = []
        for it in sorted(items, key=lambda i: (round(i.cy / 20), i.cx))[:80]:
            entry = it.to_dict()
            parsed = [c for c in parse_candidates(it.text)]
            if parsed:
                entry["coordinate"] = [
                    {"kind": c.kind, "value": c.value, "axis": c.axis, "complete": c.complete} for c in parsed
                ]
            found.append(entry)
        return _json({"texts": found, "new_in_index": added, "index_size": len(session.ocr)})

    @beta_tool
    def ocr_region(region: str, rotation: Literal[0, 90, 180, 270] = 0, upscale: float = 1.0) -> Any:
        """Lee con OCR el texto de una región y lo agrega al índice de textos del mapa.

        Úsala donde el OCR inicial no leyó una etiqueta que sí se ve en la imagen.

        Args:
            region: "x,y,w,h" en píxeles originales o celdas de la rejilla ("C4", "B2:C3").
            rotation: Giro horario del recorte antes de leerlo. 90 endereza el texto vertical que se lee de abajo hacia arriba; 270, el que se lee de arriba hacia abajo.
            upscale: Factor de ampliación previo (1 a 4). Sube a 2 o 3 para texto muy pequeño.
        """
        return run("ocr_region", _ocr_region, region=region, rotation=rotation, upscale=upscale)

    def _search_text(pattern: str, region: str, kind: str) -> str:
        rect = session.parse_region(region) if region.strip() else None
        items = session.ocr.search(pattern, rect, limit=400)
        if kind in ("utm", "geo"):
            items = [it for it in items if any(c.kind == kind for c in parse_candidates(it.text))]
        elif kind == "datum":
            items = [it for it in items if find_datum(it.text)]
        elif kind == "zone":
            items = [it for it in items if find_zone(it.text)]
        elif kind == "scale":
            items = [it for it in items if find_scale(it.text)]
        return _json({"matches": [it.to_dict() for it in items[:60]], "total": len(items)})

    @beta_tool
    def search_text(pattern: str = "", region: str = "", kind: Literal["", "utm", "geo", "datum", "zone", "scale"] = "") -> Any:
        """Busca en el texto ya reconocido por OCR, sin volver a leer la imagen.

        Args:
            pattern: Expresión regular (no distingue mayúsculas). Vacío para no filtrar por texto.
            region: Limita la búsqueda a "x,y,w,h" o a celdas de la rejilla. Vacío para todo el mapa.
            kind: Filtra por tipo: utm o geo (coordenadas), datum, zone (zona UTM) o scale (escala).
        """
        return run("search_text", _search_text, pattern=pattern, region=region, kind=kind)

    # ------------------------------------------------------------------ grilla

    def _detect_frame() -> str:
        frame = frame_from_lines(session.lines, session.width, session.height) or frame_from_contour(session.gray)
        session.frame = frame
        return _json({"frame": frame.to_dict() if frame else None})

    @beta_tool
    def detect_frame() -> Any:
        """Detecta el marco del mapa: el rectángulo donde termina la cuadrícula."""
        return run("detect_frame", _detect_frame)

    def _detect_grid(mode: str, kernel_frac: float | None, gap_frac: float | None,
                     min_support: float | None, peak_rel: float | None, cross_class: int = 0) -> str:
        info = detect_grid_impl(session, mode, kernel_frac=kernel_frac, gap_frac=gap_frac,
                                min_support=min_support, peak_rel=peak_rel, cross_class=cross_class)
        build_labels(session)
        associate(session)
        fit_grid(session)
        return _json({"info": info, "frame": session.frame.to_dict() if session.frame else None,
                      "lines": lines_brief(session), "gcps": sum(g.enabled for g in session.gcps),
                      "fit": refit(session), "warnings": session.warnings})

    @beta_tool
    def detect_grid(
        mode: Literal["lines", "ticks", "crosses"] = "lines",
        kernel_frac: float | None = None,
        gap_frac: float | None = None,
        min_support: float | None = None,
        peak_rel: float | None = None,
        cross_class: int = 0,
    ) -> Any:
        """Detecta la cuadrícula del mapa y vuelve a asociar las etiquetas. Reemplaza la grilla anterior.

        Args:
            mode: lines para líneas completas; crosses para cruces sueltas en las intersecciones; ticks para marcas cortas sobre el marco.
            kernel_frac: Largo mínimo de trazo continuo, relativo al lado menor (por defecto 0.06). Bájalo si faltan líneas cortas.
            gap_frac: Cortes que se puentean, relativo al lado menor (por defecto 0). Súbelo (0.01) para líneas de trazos o interrumpidas por texto.
            min_support: Fracción mínima de la línea con tinta (por defecto 0.35). Bájalo para líneas tenues.
            peak_rel: Largo mínimo frente a las líneas más largas (por defecto 0.35).
            cross_class: Solo en modo crosses. Un mapa puede traer dos familias de cruces de distinto tamaño (por ejemplo, retícula geográfica y cuadrícula UTM). 0 usa la más numerosa, 1 la siguiente. El resultado lista las familias en "size_classes".
        """
        return run("detect_grid", _detect_grid, mode=mode, kernel_frac=kernel_frac, gap_frac=gap_frac,
                   min_support=min_support, peak_rel=peak_rel, cross_class=cross_class)

    def _view_grid(region: str, max_side: int, show: str) -> list[dict]:
        layers = {s.strip().lower() for s in show.split(",") if s.strip()} & OVERLAY_LAYERS or {"lines", "values"}
        blocks = _view(region, max_side, "none", False, layers)
        meta = json.loads(blocks[0]["text"])
        meta["lines"] = lines_brief(session)
        meta["legend"] = "azul: verticales; rojo: horizontales; gris: fuera de retícula; naranja: residuo ampliado 25 veces"
        blocks[0]["text"] = _json(meta)
        return blocks

    @beta_tool
    def view_grid(region: str = "", max_side: int = 1568, show: str = "lines,values,labels,gcps,residuals") -> Any:
        """Muestra la grilla reconstruida dibujada sobre el mapa: cada línea con su ID y el valor asignado.

        Sirve para comprobar que las líneas detectadas coinciden con las impresas y que cada
        valor corresponde a la etiqueta que se lee en el margen.

        Args:
            region: Vacío para todo el mapa; "x,y,w,h" o celdas de la rejilla para ampliar.
            max_side: Lado largo de la vista en píxeles (512 a 2576).
            show: Capas separadas por coma: lines, values, labels (cajas del OCR: verde aceptada, rojo incoherente, amarillo sin asignar), gcps, residuals.
        """
        return run("view_grid", _view_grid, region=region, max_side=max_side, show=show)

    # ------------------------------------------------------------ coordenadas

    def _get_label_candidates(side: str) -> str:
        labels = [lab for lab in session.labels if not side or lab.side == side]
        labels.sort(key=lambda lab: (lab.side, lab.cx if lab.side in ("top", "bottom") else lab.cy))
        return _json({"labels": [lab.to_dict() for lab in labels[:100]], "total": len(labels)})

    @beta_tool
    def get_label_candidates(side: Literal["", "top", "bottom", "left", "right", "inside", "corner"] = "") -> Any:
        """Lista las etiquetas de coordenada interpretadas a partir del OCR, con la línea a la que se asociaron.

        Args:
            side: Margen del mapa a consultar. Vacío para todos.
        """
        return run("get_label_candidates", _get_label_candidates, side=side)

    def _assign_labels(assignments: list[Assignment], kind: str, step_x: float, step_y: float) -> str:
        steps = {a: s for a, s in (("x", step_x), ("y", step_y)) if s and s > 0}
        manual: list[tuple[str, float]] = []
        for item in assignments:
            ln = session.line(item["line_id"])
            if ln is None:
                raise ValueError(f"No existe la línea {item['line_id']}.")
            if str(item["value"]).strip().lower() in ("", "none", "null", "-"):
                ln.value, ln.value_source = None, "agent"  # excluida a propósito
            else:
                manual.append((ln.id, parse_value(item["value"])))
        for line_id, value in manual:
            ln = session.line(line_id)
            ln.value, ln.value_source, ln.on_lattice = value, "agent", True
        if not session.labels:
            build_labels(session)
        if not assignments:
            associate(session)
        fit_grid(session, kind or (session.label_kind if manual else None) or None, steps)
        return _json({
            "label_kind": session.label_kind,
            "axis_fits": {a: f.to_dict() for a, f in session.axis_fits.items()},
            "lines": lines_brief(session),
            "outlier_labels": [lab.to_dict() for lab in session.labels if lab.status == "outlier"][:12],
            "gcps": sum(g.enabled for g in session.gcps),
            "fit": refit(session),
            "warnings": session.warnings,
        })

    @beta_tool
    def assign_labels(assignments: list[Assignment] | None = None, kind: Literal["", "utm", "geo"] = "",
                      step_x: float = 0.0, step_y: float = 0.0) -> Any:
        """Asigna valores de coordenada a las líneas y reconstruye la grilla completa.

        Sin argumentos, asocia cada etiqueta a su línea más cercana, ajusta la progresión de
        valores, descarta lecturas incoherentes y deduce el valor de las líneas sin etiqueta.
        Con `assignments` fijas tú el valor de líneas concretas (manda sobre el OCR) y el resto
        se propaga. Con dos líneas fijadas por eje basta para definir toda la grilla.

        Args:
            assignments: Lista de {"line_id": "V3", "value": "320000"}. El valor puede ser un número (metros o grados decimales, con signo) o grados-minutos-segundos como "76°30' W". Usa "none" para excluir una línea que no es de la cuadrícula.
            kind: Tipo de coordenadas: utm (metros) o geo (grados). Vacío para deducirlo.
            step_x: Intervalo entre líneas verticales, si lo conoces (metros o grados). 0 para deducirlo.
            step_y: Intervalo entre líneas horizontales. 0 para deducirlo.
        """
        return run("assign_labels", _assign_labels, assignments=assignments or [], kind=kind,
                   step_x=step_x, step_y=step_y)

    # -------------------------------------------------------------------- CRS

    def _rank_crs(candidates: list[int]) -> str:
        proposed = propose_crs(session, settings.region_hint, settings.default_datum)
        codes = candidates or [c["epsg"] for c in proposed]
        legend = read_legend(session)
        out: dict[str, Any] = {
            "legend": {k: legend[k] for k in ("datum", "zone", "south", "scale")},
            "evidence": session.crs_evidence,
            "current": {"epsg": session.crs_epsg, "name": crs_name(session.crs_epsg), "confident": session.crs_confident},
            "label_kind": session.label_kind,
        }
        if codes and sum(g.enabled for g in session.gcps) >= 4:
            out["ranking"] = rank_crs_impl(session, codes)
            if session.label_kind != "geo":
                out["note"] = ("Las etiquetas son UTM: todas las zonas ajustan igual. La zona y el datum "
                               "solo pueden salir de la leyenda o de otra evidencia del mapa.")
        else:
            out["candidates"] = proposed
        return _json(out)

    @beta_tool
    def rank_crs(candidates: list[int] | None = None) -> Any:
        """Resume la evidencia sobre el sistema de coordenadas y compara candidatos por error de ajuste.

        Args:
            candidates: Códigos EPSG a comparar. Vacío para usar los que se deducen de la leyenda y la configuración.
        """
        return run("rank_crs", _rank_crs, candidates=candidates or [])

    def _set_crs(epsg: int, evidence: str) -> str:
        name = crs_name(epsg)
        if "desconocido" in name:
            raise ValueError(f"EPSG:{epsg} no existe.")
        if not evidence.strip():
            raise ValueError("Indica la evidencia: qué texto del mapa respalda este sistema de coordenadas.")
        session.crs_epsg = int(epsg)
        session.crs_confident = True
        session.crs_evidence.append(f"agente: {evidence.strip()}")
        return _json({"epsg": epsg, "name": name, "fit": refit(session)})

    @beta_tool
    def set_crs(epsg: int, evidence: str) -> Any:
        """Fija el sistema de coordenadas del mapa. Hazlo solo con evidencia leída en el propio mapa.

        Args:
            epsg: Código EPSG, por ejemplo 32718 (WGS 84 / UTM zona 18S) o 24878 (PSAD56 / UTM zona 18S).
            evidence: Texto del mapa que lo respalda y dónde está, por ejemplo: leyenda inferior "Datum WGS 84, Zona 18 Sur".
        """
        return run("set_crs", _set_crs, epsg=epsg, evidence=evidence)

    # ----------------------------------------------------------------- georef

    def _edit_gcps(action: str, gcp_id: str, px: float, py: float, x: str, y: str, snap: bool) -> str:
        if action == "list":
            gcps = sorted(session.gcps, key=lambda g: -(g.residual_px or 0))
            return _json({"gcps": [g.to_dict() for g in gcps[:80]], "total": len(gcps),
                          "enabled": sum(g.enabled for g in gcps)})
        if action in ("disable", "enable"):
            wanted = gcp_id.strip().upper()
            # Un ID de línea ("V3") afecta a todos los puntos de esa línea.
            hits = [g for g in session.gcps if g.id.upper() == wanted or
                    re.fullmatch(rf"{re.escape(wanted)}H\d+|V\d+{re.escape(wanted)}", g.id.upper())]
            if not hits:
                raise ValueError(f"No hay puntos de control con ID {gcp_id}.")
            for g in hits:
                g.enabled = action == "enable"
            line = session.line(wanted)
            if line is not None and action == "disable":
                # Descartar una línea le quita también el valor: deja de contar en la grilla.
                line.value, line.value_source = None, "agent"
            return _json({"changed": [g.id for g in hits], "fit": refit(session)})
        if action == "add":
            sx, sy = (snap_point(session.gray, px, py) if snap else (px, py))
            gid = f"M{sum(g.source == 'agent' for g in session.gcps) + 1}"
            session.gcps.append(GCP(gid, sx, sy, parse_value(x), parse_value(y), "agent"))
            return _json({"added": gid, "pixel": [round(sx, 2), round(sy, 2)],
                          "moved_px": round(float(np.hypot(sx - px, sy - py)), 2), "fit": refit(session)})
        raise ValueError(f"Acción desconocida: {action}")

    @beta_tool
    def edit_gcps(action: Literal["list", "add", "disable", "enable"], gcp_id: str = "",
                  px: float = 0.0, py: float = 0.0, x: str = "", y: str = "", snap: bool = True) -> Any:
        """Consulta o modifica los puntos de control. Tras cada cambio se reajusta la transformación.

        Args:
            action: list (ordenados por residuo), add (punto manual), disable o enable.
            gcp_id: Para disable/enable: ID del punto ("V3H5") o de una línea ("V3") para todos sus puntos.
            px: Para add: columna aproximada en píxeles originales.
            py: Para add: fila aproximada en píxeles originales.
            x: Para add: Este o longitud del punto (número, o grados como "76°30' W").
            y: Para add: Norte o latitud del punto.
            snap: Para add: ajusta el punto a la esquina o cruce de trazos más cercano.
        """
        return run("edit_gcps", _edit_gcps, action=action, gcp_id=gcp_id, px=px, py=py, x=x, y=y, snap=snap)

    def _fit_transform(kind: str) -> str:
        if not any(g.source == "grid" for g in session.gcps):
            build_gcps(session)
        fit_transform_impl(session, kind)
        return _json(fit_brief(session))

    @beta_tool
    def fit_transform(kind: Literal["auto", "affine", "projective", "poly2", "tps"] = "auto") -> Any:
        """Ajusta la transformación píxel-mundo con los puntos de control habilitados.

        Devuelve el error (RMSE en píxeles y metros), la validación cruzada dejando un punto
        fuera y los puntos con mayor residuo.

        Args:
            kind: auto elige afín salvo que otro modelo mejore con claridad; affine para escaneos planos; projective para fotos con perspectiva; poly2 o tps para papel deformado.
        """
        return run("fit_transform", _fit_transform, kind=kind)

    def _validate() -> str:
        return _json(validate_impl(session, settings))

    @beta_tool
    def validate() -> Any:
        """Ejecuta los chequeos de coherencia del resultado: puntos, error, orientación, escala, monotonía y CRS."""
        return run("validate", _validate)

    def _submit_result(status: str, epsg: int, confidence: float, notes: str) -> str:
        if epsg and status in ("ok", "needs_review") and epsg != session.crs_epsg:
            raise ValueError(f"El EPSG indicado ({epsg}) no coincide con el fijado en la sesión "
                             f"({session.crs_epsg}). Usa set_crs antes de cerrar.")
        session.result = {"status": status, "epsg": epsg or session.crs_epsg,
                          "confidence": float(min(1.0, max(0.0, confidence))), "notes": notes.strip()}
        return _json({"received": True, "fit": fit_brief(session)})

    @beta_tool
    def submit_result(status: Literal["ok", "needs_review", "no_coordinates", "failed"], epsg: int,
                      confidence: float, notes: str) -> Any:
        """Cierra el trabajo sobre el mapa. Llámala una sola vez, al final.

        Args:
            status: ok si la georreferenciación es confiable; needs_review si existe pero hay dudas (CRS sin evidencia, residuos altos, pocas etiquetas); no_coordinates si el mapa no trae coordenadas; failed si no fue posible.
            epsg: Código EPSG del sistema de coordenadas, o 0 si no se pudo determinar.
            confidence: Confianza de 0 a 1 en que el resultado es correcto.
            notes: Qué se verificó, qué evidencia respalda el CRS y qué queda en duda. Dos o tres frases.
        """
        return run("submit_result", _submit_result, status=status, epsg=epsg, confidence=confidence, notes=notes)

    return [view_image, ocr_region, search_text, detect_frame, detect_grid, view_grid, get_label_candidates,
            assign_labels, rank_crs, set_crs, edit_gcps, fit_transform, validate, submit_result]
