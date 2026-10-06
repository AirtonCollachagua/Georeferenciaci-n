"""Evaluación del pipeline contra verdad conocida."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..georef.transform import to_metres
from ..pipeline import decide_status, open_session, process_session, run_prepass
from ..session import MapSession
from .metrics import error_vs_truth
from .synth import FakeOcr, SynthSpec, geo_spec, make_map

# (nombre, especificación del mapa, opciones del OCR simulado)
SYNTHETIC_SUITE: list[tuple[str, SynthSpec, dict[str, Any]]] = [
    ("utm_lineas", SynthSpec(rotation_deg=1.3, seed=1), {}),
    ("utm_lineas_recto", SynthSpec(seed=2), {}),
    ("utm_etiquetas_mE", SynthSpec(rotation_deg=-2.5, seed=3, label_format="unit"), {}),
    ("utm_ocr_ruidoso", SynthSpec(rotation_deg=0.6, seed=4), {"drop": 0.3, "corrupt": 0.2, "seed": 7}),
    ("utm_texto_vertical", SynthSpec(seed=5), {"vertical_needs_rotation": True}),
    ("utm_grande_7000px", SynthSpec(width=7000, height=5200, margin=300, res=2.0, rotation_deg=0.7, seed=8), {}),
    ("geo_lineas", geo_spec(rotation_deg=-0.8, seed=2), {}),
    ("geo_hemisferio", geo_spec(rotation_deg=0.4, seed=6, hemisphere_letters=True), {}),
    ("utm_cruces", SynthSpec(style="crosses", rotation_deg=0.9, seed=9), {}),
    ("utm_marcas", SynthSpec(style="ticks", rotation_deg=-0.7, seed=11), {}),
    ("geo_marcas", geo_spec(style="ticks", rotation_deg=0.5, seed=12), {}),
    ("sin_coordenadas", SynthSpec(style="none", seed=13), {"drop": 1.0}),
]


def run_case(name: str, spec: SynthSpec, ocr_options: dict[str, Any], settings: Settings) -> dict[str, Any]:
    synth = make_map(spec)
    session = MapSession(synth.image, f"{name}.png")
    session.ocr_provider = FakeOcr(synth, **ocr_options)
    started = time.time()
    summary = run_prepass(session, settings)
    decision = decide_status(session, settings)
    row: dict[str, Any] = {
        "case": name,
        "seconds": round(time.time() - started, 2),
        "mode": summary["grid"].get("mode"),
        "labels_ok": f"{summary['labels']['ok']}/{summary['labels']['total']}",
        "gcps": summary["gcps"],
        "epsg": session.crs_epsg,
        "status": decision["status"],
        "expected_epsg": spec.epsg if spec.style != "none" else None,
    }
    if session.fit:
        row.update(fit=session.fit.kind, **{k: round(v, 3) for k, v in error_vs_truth(session, synth.pixel_to_world).items()})
    return row


def run_synthetic_suite(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or Settings()
    return [run_case(name, spec, ocr, settings) for name, spec, ocr in SYNTHETIC_SUITE]


def format_table(rows: list[dict[str, Any]]) -> str:
    columns = ["case", "status", "mode", "labels_ok", "gcps", "fit", "rmse_px", "max_px", "rmse_m", "epsg", "seconds"]
    table = [columns] + [[str(r.get(c, "-")) for c in columns] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in table]
    return "\n".join([lines[0], "  ".join("-" * w for w in widths)] + lines[1:])


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Lee un .points de QGIS: (píxeles, coordenadas, EPSG si el archivo lo declara)."""
    epsg: int | None = None
    px, world = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("#CRS:"):
            try:
                from pyproj import CRS

                epsg = CRS.from_user_input(line[5:].strip()).to_epsg()
            except Exception:
                epsg = None
            continue
        parts = line.split(",")
        if len(parts) < 4 or not parts[0].replace(".", "").replace("-", "").strip().isdigit():
            continue
        if len(parts) > 4 and parts[4].strip() == "0":
            continue  # punto deshabilitado
        world.append((float(parts[0]), float(parts[1])))
        px.append((float(parts[2]), -float(parts[3])))  # QGIS guarda la fila en negativo
    return np.array(px), np.array(world), epsg


def run_truth_dir(folder: Path, settings: Settings, ocr_provider: Any = None, client: Any = None,
                  out_root: Path = Path("out/eval")) -> list[dict[str, Any]]:
    """Procesa cada mapa que tenga un .points de referencia al lado y mide el error en esos puntos."""
    from ..imaging.loader import SUPPORTED_SUFFIXES

    rows = []
    for image in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES):
        reference = next((c for c in (image.with_suffix(".points"), Path(str(image) + ".points")) if c.exists()), None)
        if reference is None:
            continue
        px, world, truth_epsg = read_points(reference)
        session = open_session(image, settings, ocr_provider)
        doc = process_session(session, out_root / session.name, settings, client, use_agent=client is not None)
        row: dict[str, Any] = {"case": image.name, "status": doc["status"], "mode": session.grid_info.get("mode"),
                               "labels_ok": f"{sum(l.status == 'ok' for l in session.labels)}/{len(session.labels)}",
                               "gcps": sum(g.enabled for g in session.gcps), "epsg": session.crs_epsg,
                               "expected_epsg": truth_epsg, "seconds": doc.get("seconds")}
        fit = session.fit
        if fit is not None and len(px):
            expected = world
            if truth_epsg and fit.epsg and truth_epsg != fit.epsg:
                from pyproj import Transformer

                x, y = Transformer.from_crs(truth_epsg, fit.epsg, always_xy=True).transform(world[:, 0], world[:, 1])
                expected = np.column_stack([x, y])
            metres = to_metres(fit.pixel_to_world(px) - expected, expected, fit.units == "deg")
            pixels = np.hypot(*(fit.world_to_pixel(expected) - px).T)
            row.update(fit=fit.kind, rmse_px=round(float(np.sqrt(np.mean(pixels**2))), 3),
                       max_px=round(float(pixels.max()), 3), rmse_m=round(float(np.sqrt(np.mean(metres**2))), 3))
        rows.append(row)
    return rows
