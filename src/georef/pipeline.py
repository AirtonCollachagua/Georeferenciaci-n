"""Flujo por mapa: ingesta -> pre-análisis determinista -> agente -> exportación."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import tracing
from .config import Settings
from .coords.crs import crs_name, propose_crs
from .coords.labels import associate, build_labels, drop_inconsistent_lines, fit_grid
from .georef.export import export_all
from .georef.transform import fit_transform, rank_crs
from .georef.validate import validate
from .grid import detect_grid
from .imaging.loader import load_image
from .ocr.tiling import ocr_full, ocr_side_bands
from .session import MapSession


@tracing.traced("ocr_tiles", run_type="tool")
def _ocr_stage(session: MapSession) -> dict[str, Any]:
    added = ocr_full(session)
    added += ocr_side_bands(session)
    provider = session.ocr_provider
    return {"texts": len(session.ocr), "new": added,
            "calls": getattr(provider, "calls", None), "cache_hits": getattr(provider, "cache_hits", None)}


GRID_MODES = ("lines", "crosses", "ticks")


@tracing.traced("detect_grid", run_type="tool")
def _grid_stage(session: MapSession, mode: str, **overrides: Any) -> dict[str, Any]:
    return detect_grid(session, mode, **overrides)


def _label_grid(session: MapSession) -> bool:
    """Asocia etiquetas y valores. La cuadrícula sirve si ambos ejes quedan resueltos."""
    build_labels(session)
    associate(session)
    fit_grid(session)
    return len(session.axis_fits) == 2 and sum(g.enabled for g in session.gcps) >= 4


def _choose_crs(session: MapSession, candidates: list[dict[str, Any]]) -> None:
    """Un mapa rotulado en grados puede estar dibujado en una proyección: gana la que
    deja el mapa más cerca de un afín, y solo si la ventaja es clara."""
    enabled = sum(g.enabled for g in session.gcps)
    if session.label_kind != "geo" or len(candidates) < 2 or enabled < 4:
        return
    ranking = rank_crs(session, [c["epsg"] for c in candidates])
    by_epsg = {r["epsg"]: r for r in ranking}
    for c in candidates:
        c.update({k: v for k, v in by_epsg.get(c["epsg"], {}).items() if k != "name"})
    geographic, best = by_epsg.get(candidates[0]["epsg"], {}), ranking[0]
    if "rmse_px" in best and "rmse_px" in geographic and best["rmse_px"] < 0.85 * geographic["rmse_px"]:
        session.crs_epsg = best["epsg"]
        session.crs_evidence.append(f"el mapa ajusta mejor en {best['name']} que en coordenadas geográficas")


@tracing.traced("auto_fit")
def _fit_stage(session: MapSession, settings: Settings) -> dict[str, Any]:
    """Prueba líneas completas; si no resuelven el mapa, cruces y luego marcas del marco."""
    resolved = _label_grid(session)
    attempts = [("crosses", {"cross_class": 0}), ("crosses", {"cross_class": 1}), ("ticks", {})]
    for mode, overrides in attempts:
        if resolved:
            break
        info = _grid_stage(session, mode, **overrides)
        if overrides.get("cross_class", 0) >= max(1, len(info.get("size_classes", []))):
            continue  # no hay una segunda familia de cruces
        resolved = _label_grid(session)
    if not resolved and session.grid_info.get("mode") != "lines":
        # Ningún modo bastó: el agente parte de las líneas, que es lo más informativo.
        _grid_stage(session, "lines")
        _label_grid(session)

    candidates = propose_crs(session, settings.region_hint, settings.default_datum)
    _choose_crs(session, candidates)
    dropped: list[str] = []
    if sum(g.enabled for g in session.gcps) >= 3:
        fit_transform(session, "auto")
        dropped = drop_inconsistent_lines(session)
    return {"crs_candidates": candidates, "gcps": sum(g.enabled for g in session.gcps), "dropped_lines": dropped}


@tracing.traced("prepass")
def run_prepass(session: MapSession, settings: Settings) -> dict[str, Any]:
    """Todo lo que se puede resolver sin modelo. Devuelve el resumen que recibe el agente."""
    _grid_stage(session, "lines")
    if session.ocr_provider is not None:
        _ocr_stage(session)
    else:
        session.warnings.append("Sin proveedor de OCR: no se leyeron etiquetas.")
    stage = _fit_stage(session, settings)
    return summarize(session, settings, stage["crs_candidates"])


def summarize(session: MapSession, settings: Settings, crs_candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Estado compacto del mapa: lo que ve el agente y lo que se guarda en el reporte."""
    labels = session.labels
    by_side: dict[str, int] = {}
    for lab in labels:
        by_side[lab.side or "?"] = by_side.get(lab.side or "?", 0) + 1
    summary: dict[str, Any] = {
        "image": {"width": session.width, "height": session.height, "dpi": session.dpi},
        "ref_grid": session.ref_grid.describe(),
        "frame": session.frame.to_dict() if session.frame else None,
        "grid": {
            **session.grid_info,
            "vertical_lines": len(session.lines_of("v")),
            "horizontal_lines": len(session.lines_of("h")),
            "lines_with_value": sum(ln.value is not None for ln in session.lines),
        },
        "ocr_texts": len(session.ocr),
        "labels": {
            "total": len(labels),
            "by_side": by_side,
            "ok": sum(lab.status == "ok" for lab in labels),
            "outliers": [lab.to_dict() for lab in labels if lab.status == "outlier"][:12],
            "abbreviated": sum(not lab.complete for lab in labels),
        },
        "label_kind": session.label_kind,
        "axis_fits": {a: f.to_dict() for a, f in session.axis_fits.items()},
        "crs": {"epsg": session.crs_epsg, "name": crs_name(session.crs_epsg),
                "confident": session.crs_confident, "evidence": session.crs_evidence},
        "gcps": sum(g.enabled for g in session.gcps),
        "fit": session.fit.summary() if session.fit else None,
        "warnings": session.warnings,
    }
    if crs_candidates is not None:
        summary["crs"]["candidates"] = crs_candidates
    if session.fit:
        summary["validation"] = validate(session, settings)
    return summary


def decide_status(session: MapSession, settings: Settings) -> dict[str, Any]:
    """Estado final. El agente propone; los chequeos tienen la última palabra sobre un "ok"."""
    agent = session.result or {}
    notes = [agent["notes"]] if agent.get("notes") else []
    has_coordinates = any(lab.complete for lab in session.labels) or bool(session.gcps)

    if session.fit is None:
        if agent.get("status") in ("no_coordinates", "failed"):
            status = agent["status"]
        elif session.ocr_provider is None and not agent:
            status = "needs_review"
            notes.append("No se leyó el texto del mapa (OCR sin configurar): no se puede saber si trae coordenadas.")
        elif not has_coordinates:
            status = "no_coordinates"
            notes.append("No se encontraron coordenadas legibles en el mapa.")
        else:
            status = "needs_review"
            notes.append("Hay coordenadas, pero no alcanzaron para ajustar una transformación.")
        return {"status": status, "confidence": float(agent.get("confidence", 0.0)), "notes": notes, "validation": None}

    report = validate(session, settings)
    status = agent.get("status") or report["suggested_status"]
    if status == "ok" and not report["passed"]:
        failed = [c["check"] for c in report["checks"] if c["critical"] and not c["ok"]]
        status = "needs_review"
        notes.append("Rebajado a revisión por los chequeos: " + ", ".join(failed) + ".")
    confidence = agent.get("confidence")
    if confidence is None:
        confidence = 0.9 if report["passed"] else 0.5 if session.crs_epsg else 0.3
    return {"status": status, "confidence": float(confidence), "notes": notes, "validation": report}


def open_session(path: str | Path, settings: Settings, ocr_provider: Any = None) -> MapSession:
    image, dpi = load_image(path)
    session = MapSession(image, path, dpi)
    session.ocr_provider = ocr_provider
    return session


@tracing.traced("georef_map")
def process_session(
    session: MapSession,
    out_dir: str | Path,
    settings: Settings,
    client: Any = None,
    use_agent: bool = True,
) -> dict[str, Any]:
    """Procesa un mapa ya cargado y escribe sus entregables. Devuelve el contenido de georef.json."""
    started = time.time()
    out_dir = Path(out_dir)
    tracing.set_metadata(map=session.path.name, map_sha=session.sha, model=settings.model,
                         effort=settings.effort, agent=bool(use_agent and client))
    prepass = run_prepass(session, settings)

    agent_report: dict[str, Any] = {"used": False}
    if use_agent and client is not None:
        from .agent.runner import run_agent

        agent_report = run_agent(session, settings, client, prepass, out_dir)

    if session.fit is None and sum(g.enabled for g in session.gcps) >= 3:
        fit_transform(session, "auto")
    decision = decide_status(session, settings)
    report = {
        **decision,
        "agent": agent_report,
        "seconds": round(time.time() - started, 1),
        "trace": tracing.current_run_info(),
    }
    doc = _export(session, out_dir, report)
    tracing.send_feedback(doc)
    return doc


@tracing.traced("export")
def _export(session: MapSession, out_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    return export_all(session, out_dir, report)


def process_map(path: str | Path, out_root: str | Path, settings: Settings,
                ocr_provider: Any = None, client: Any = None, use_agent: bool = True) -> dict[str, Any]:
    """Carga un mapa y lo procesa. Los entregables quedan en `out_root/<nombre>/`."""
    session = open_session(path, settings, ocr_provider)
    return process_session(session, Path(out_root) / session.name, settings, client, use_agent)
