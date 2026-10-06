"""Ajuste píxel <-> mundo a partir de los puntos de control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from pyproj import CRS

from ..coords.crs import world_coords

if TYPE_CHECKING:
    from ..session import MapSession

KINDS = ("auto", "affine", "projective", "poly2", "tps")


class _Norm:
    """Centra y escala: las coordenadas UTM (millones) condicionan mal los ajustes."""

    def __init__(self, pts: np.ndarray):
        self.mean = pts.mean(axis=0)
        self.scale = float(np.abs(pts - self.mean).mean()) or 1.0

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.mean) / self.scale

    def undo(self, pts: np.ndarray) -> np.ndarray:
        return pts * self.scale + self.mean


class _Affine:
    min_points = 3

    def __init__(self, px: np.ndarray, world: np.ndarray):
        self.np_, self.nw = _Norm(px), _Norm(world)
        a = np.column_stack([self.np_.apply(px), np.ones(len(px))])
        self.m, *_ = np.linalg.lstsq(a, self.nw.apply(world), rcond=None)

    def forward(self, px: np.ndarray) -> np.ndarray:
        a = np.column_stack([self.np_.apply(px), np.ones(len(px))])
        return self.nw.undo(a @ self.m)

    def inverse(self, world: np.ndarray) -> np.ndarray:
        w = self.nw.apply(world) - self.m[2]
        return self.np_.undo(w @ np.linalg.inv(self.m[:2]))


class _Projective:
    min_points = 4

    def __init__(self, px: np.ndarray, world: np.ndarray):
        self.np_, self.nw = _Norm(px), _Norm(world)
        h, _ = cv2.findHomography(self.np_.apply(px), self.nw.apply(world), 0)
        if h is None:
            raise ValueError("No se pudo ajustar la homografía.")
        self.h, self.h_inv = h, np.linalg.inv(h)

    @staticmethod
    def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
        hom = np.column_stack([pts, np.ones(len(pts))]) @ h.T
        return hom[:, :2] / hom[:, 2:3]

    def forward(self, px: np.ndarray) -> np.ndarray:
        return self.nw.undo(self._apply(self.h, self.np_.apply(px)))

    def inverse(self, world: np.ndarray) -> np.ndarray:
        return self.np_.undo(self._apply(self.h_inv, self.nw.apply(world)))


class _Poly2:
    min_points = 7

    def __init__(self, px: np.ndarray, world: np.ndarray):
        self.np_, self.nw = _Norm(px), _Norm(world)
        p, w = self.np_.apply(px), self.nw.apply(world)
        self.c, *_ = np.linalg.lstsq(self._terms(p), w, rcond=None)
        self.d, *_ = np.linalg.lstsq(self._terms(w), p, rcond=None)

    @staticmethod
    def _terms(p: np.ndarray) -> np.ndarray:
        x, y = p[:, 0], p[:, 1]
        return np.column_stack([np.ones(len(p)), x, y, x * x, x * y, y * y])

    def forward(self, px: np.ndarray) -> np.ndarray:
        return self.nw.undo(self._terms(self.np_.apply(px)) @ self.c)

    def inverse(self, world: np.ndarray) -> np.ndarray:
        return self.np_.undo(self._terms(self.nw.apply(world)) @ self.d)


class _Tps:
    min_points = 5

    def __init__(self, px: np.ndarray, world: np.ndarray):
        from scipy.interpolate import RBFInterpolator

        self.np_, self.nw = _Norm(px), _Norm(world)
        p, w = self.np_.apply(px), self.nw.apply(world)
        self.f = RBFInterpolator(p, w, kernel="thin_plate_spline")
        self.g = RBFInterpolator(w, p, kernel="thin_plate_spline")

    def forward(self, px: np.ndarray) -> np.ndarray:
        return self.nw.undo(self.f(self.np_.apply(px)))

    def inverse(self, world: np.ndarray) -> np.ndarray:
        return self.np_.undo(self.g(self.nw.apply(world)))


_MODELS = {"affine": _Affine, "projective": _Projective, "poly2": _Poly2, "tps": _Tps}


@dataclass
class FitResult:
    kind: str
    epsg: int | None
    n: int
    rmse_px: float
    max_px: float
    loo_rmse_px: float | None
    rmse_m: float
    pixel_size: tuple[float, float]
    rotation_deg: float
    mirrored: bool
    units: str
    affine: np.ndarray  # 3x2: mundo = [px, py, 1] @ affine
    residual_vectors: dict[str, tuple[float, float]] = field(default_factory=dict)
    outliers: list[str] = field(default_factory=list)
    model: Any = None

    def pixel_to_world(self, px: np.ndarray) -> np.ndarray:
        return self.model.forward(np.atleast_2d(np.asarray(px, dtype=float)))

    def world_to_pixel(self, world: np.ndarray) -> np.ndarray:
        return self.model.inverse(np.atleast_2d(np.asarray(world, dtype=float)))

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "epsg": self.epsg,
            "n_gcps": self.n,
            "rmse_px": round(self.rmse_px, 3),
            "max_px": round(self.max_px, 3),
            "loo_rmse_px": None if self.loo_rmse_px is None else round(self.loo_rmse_px, 3),
            "rmse_m": round(self.rmse_m, 3),
            "pixel_size": [float(f"{v:.6g}") for v in self.pixel_size],
            "units": self.units,
            "rotation_deg": round(self.rotation_deg, 3),
            "outliers": self.outliers,
        }


def _plain_affine(px: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Afín sin normalizar, como matriz 3x2 que opera sobre píxeles crudos."""
    model = _Affine(px, world)
    origin = model.forward(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    return np.vstack([origin[1] - origin[0], origin[2] - origin[0], origin[0]])


def to_metres(delta: np.ndarray, world: np.ndarray, geographic: bool) -> np.ndarray:
    if not geographic:
        return np.hypot(delta[:, 0], delta[:, 1])
    lat = np.radians(world[:, 1])
    return np.hypot(delta[:, 0] * 111_320 * np.cos(lat), delta[:, 1] * 110_540)


def _loo(cls: type, px: np.ndarray, world: np.ndarray) -> float | None:
    n = len(px)
    if n <= cls.min_points + 1:
        return None
    idx = np.arange(n) if n <= 150 else np.linspace(0, n - 1, 150).astype(int)
    errors = []
    for i in idx:
        keep = np.arange(n) != i
        try:
            model = cls(px[keep], world[keep])
            errors.append(np.hypot(*(model.inverse(world[i : i + 1])[0] - px[i])))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def _evaluate(kind: str, ids: list[str], px: np.ndarray, world: np.ndarray, epsg: int | None, geographic: bool) -> FitResult:
    cls = _MODELS[kind]
    if len(px) < cls.min_points:
        raise ValueError(f"El ajuste {kind} necesita al menos {cls.min_points} puntos; hay {len(px)}.")
    model = cls(px, world)
    residual = model.inverse(world) - px
    dist = np.hypot(residual[:, 0], residual[:, 1])
    metres = to_metres(model.forward(px) - world, world, geographic)

    affine = _plain_affine(px, world)
    sx, sy = float(np.hypot(*affine[0])), float(np.hypot(*affine[1]))
    threshold = max(1.5, 3 * float(np.median(dist)))
    return FitResult(
        kind=kind, epsg=epsg, n=len(px),
        rmse_px=float(np.sqrt(np.mean(dist**2))), max_px=float(dist.max()),
        loo_rmse_px=_loo(cls, px, world),
        rmse_m=float(np.sqrt(np.mean(metres**2))),
        pixel_size=(sx, sy),
        rotation_deg=math.degrees(math.atan2(affine[0, 1], affine[0, 0])),
        mirrored=bool(np.linalg.det(affine[:2]) > 0),
        units="deg" if geographic else "m",
        affine=affine,
        residual_vectors={i: (float(r[0]), float(r[1])) for i, r in zip(ids, residual)},
        outliers=[i for i, d in zip(ids, dist) if d > threshold],
        model=model,
    )


def fit_transform(session: "MapSession", kind: str = "auto", epsg: int | None = None) -> FitResult:
    """Ajusta la transformación con los puntos habilitados y la guarda en la sesión."""
    kind = (kind or "auto").lower()
    if kind not in KINDS:
        raise ValueError(f"Tipo de ajuste desconocido: {kind}. Opciones: {', '.join(KINDS)}")
    gcps = [g for g in session.gcps if g.enabled]
    if len(gcps) < 3:
        raise ValueError(f"Hacen falta al menos 3 puntos de control habilitados; hay {len(gcps)}.")
    epsg = epsg if epsg is not None else session.crs_epsg
    ids = [g.id for g in gcps]
    px = np.array([[g.px, g.py] for g in gcps], dtype=float)
    raw = np.array([[g.x, g.y] for g in gcps], dtype=float)
    world = world_coords(session, epsg, raw) if epsg else raw
    geographic = CRS.from_epsg(epsg).is_geographic if epsg else session.label_kind == "geo"

    if kind != "auto":
        result = _evaluate(kind, ids, px, world, epsg, geographic)
    else:
        result = _evaluate("affine", ids, px, world, epsg, geographic)
        # Un modelo más flexible solo se acepta si el afín falla de verdad y el otro
        # mejora con claridad fuera de muestra; si no, sobreajusta el ruido de las líneas.
        for candidate, min_n, min_error, gain in (("projective", 8, 1.0, 0.6), ("poly2", 20, 1.5, 0.5)):
            base = result.loo_rmse_px
            if len(px) < min_n or base is None or base <= min_error:
                break
            other = _evaluate(candidate, ids, px, world, epsg, geographic)
            if other.loo_rmse_px is not None and other.loo_rmse_px < gain * base:
                result = other

    for g in session.gcps:
        vec = result.residual_vectors.get(g.id)
        g.residual_px = float(np.hypot(*vec)) if vec else None
    session.fit = result
    return result


def rank_crs(session: "MapSession", epsgs: list[int]) -> list[dict[str, Any]]:
    """Ajusta un afín en cada CRS candidato y los ordena por error.

    Solo discrimina cuando las etiquetas son geográficas: con etiquetas UTM los
    valores ya están en la proyección y todas las zonas ajustan igual.
    """
    gcps = [g for g in session.gcps if g.enabled]
    if len(gcps) < 4:
        raise ValueError("Hacen falta al menos 4 puntos de control para comparar sistemas de coordenadas.")
    ids = [g.id for g in gcps]
    px = np.array([[g.px, g.py] for g in gcps], dtype=float)
    raw = np.array([[g.x, g.y] for g in gcps], dtype=float)
    ranking = []
    for epsg in epsgs:
        try:
            crs = CRS.from_epsg(epsg)
            fit = _evaluate("affine", ids, px, world_coords(session, epsg, raw), epsg, crs.is_geographic)
        except Exception as exc:
            ranking.append({"epsg": epsg, "error": str(exc)[:160]})
            continue
        ranking.append({"epsg": epsg, "name": crs.name, "rmse_px": round(fit.rmse_px, 3),
                        "loo_rmse_px": None if fit.loo_rmse_px is None else round(fit.loo_rmse_px, 3)})
    ranking.sort(key=lambda r: r.get("rmse_px", float("inf")))
    return ranking
