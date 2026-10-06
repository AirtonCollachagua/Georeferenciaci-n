"""Chequeos de coherencia del resultado: la última barrera contra un "ok" falso."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..coords.crs import area_of_use_ok, read_legend

if TYPE_CHECKING:
    from ..config import Settings
    from ..session import MapSession


def _check(name: str, ok: bool, detail: str, critical: bool = True) -> dict[str, Any]:
    return {"check": name, "ok": bool(ok), "critical": critical, "detail": detail}


def validate(session: "MapSession", settings: "Settings") -> dict[str, Any]:
    """Devuelve los chequeos y el estado que el sistema está dispuesto a firmar."""
    checks: list[dict[str, Any]] = []
    fit = session.fit
    if fit is None:
        return {"checks": [_check("ajuste", False, "No hay transformación ajustada.")],
                "suggested_status": "needs_review", "passed": False}

    checks.append(_check("puntos de control", fit.n >= settings.ok_min_gcps,
                         f"{fit.n} puntos (mínimo {settings.ok_min_gcps})"))
    checks.append(_check("error de ajuste", fit.rmse_px <= settings.ok_rmse_px,
                         f"RMSE {fit.rmse_px:.2f} px (umbral {settings.ok_rmse_px})"))
    if fit.loo_rmse_px is not None:
        checks.append(_check("validación cruzada", fit.loo_rmse_px <= 1.5 * settings.ok_rmse_px,
                             f"RMSE dejando uno fuera {fit.loo_rmse_px:.2f} px"))
    checks.append(_check("orientación", not fit.mirrored,
                         "imagen en espejo respecto al terreno" if fit.mirrored else "norte y este coherentes"))
    checks.append(_check("rotación", abs(fit.rotation_deg) < 10, f"{fit.rotation_deg:.2f}°", critical=False))

    sx, sy = fit.pixel_size
    if fit.units == "m":
        ratio = sx / sy if sy else 0
        checks.append(_check("píxel cuadrado", abs(ratio - 1) < 0.03,
                             f"{sx:.4g} x {sy:.4g} m/px (relación {ratio:.4f})", critical=False))
        scale = read_legend(session)["scale"]
        if scale and session.dpi:
            expected = scale * 0.0254 / session.dpi
            ok = abs((sx + sy) / 2 / expected - 1) < 0.05
            checks.append(_check("escala declarada", ok,
                                 f"1:{scale} a {session.dpi:.0f} DPI implica {expected:.4g} m/px; "
                                 f"medido {(sx + sy) / 2:.4g}", critical=False))

    for axis, family, increasing in (("x", "v", True), ("y", "h", False)):
        lines = [ln for ln in session.lines_of(family) if ln.value is not None]
        if len(lines) >= 2:
            values = np.array([ln.value for ln in sorted(lines, key=lambda l: l.pos(session.width, session.height))])
            diffs = np.diff(values)
            ok = bool((diffs > 0).all() if increasing else (diffs < 0).all())
            sense = "crecen hacia la derecha" if increasing else "decrecen hacia abajo"
            checks.append(_check(f"monotonía eje {axis}", ok, f"los valores {'sí' if ok else 'no'} {sense}"))

    if fit.epsg:
        centre = fit.pixel_to_world([[session.width / 2, session.height / 2]])[0]
        ok, detail = area_of_use_ok(fit.epsg, float(centre[0]), float(centre[1]))
        checks.append(_check("área de uso del CRS", ok, detail))
    else:
        checks.append(_check("CRS", False, "No se ha fijado un sistema de coordenadas."))
    checks.append(_check("evidencia del CRS", session.crs_confident,
                         "; ".join(session.crs_evidence) or "sin evidencia"))

    passed = all(c["ok"] for c in checks if c["critical"])
    return {"checks": checks, "passed": passed, "suggested_status": "ok" if passed else "needs_review"}
