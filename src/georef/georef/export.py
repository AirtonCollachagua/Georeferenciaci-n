"""Entregables: GeoTIFF, puntos de control, world file, JSON y lámina de control de calidad."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from ..coords.crs import crs_name
from ..imaging.annotate import render_view

if TYPE_CHECKING:
    from ..session import MapSession
    from .transform import FitResult

_WORLD_EXT = {".tif": ".tfw", ".tiff": ".tfw", ".jpg": ".jgw", ".jpeg": ".jgw", ".png": ".pgw", ".bmp": ".bpw"}
# cv2.remap y warpPerspective no admiten lados de 32768 píxeles o más.
_MAX_WARP_SIDE = 32000


def _write_bytes(path: Path, ext: str, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"No se pudo escribir {path}")
    buf.tofile(str(path))


def _is_axis_aligned(fit: "FitResult") -> bool:
    # 0,002° desplazan menos de medio píxel en 10 000 px: por debajo no vale la pena remuestrear.
    return fit.kind == "affine" and abs(fit.rotation_deg) < 0.002


def _output_grid(session: "MapSession", fit: "FitResult") -> tuple[float, float, float, int, int]:
    """Lienzo con el norte arriba que contiene la imagen: (x_min, y_max, resolución, ancho, alto)."""
    w, h = session.width, session.height
    edge = np.linspace(0, 1, 9)
    border = np.array(
        [(t * w, 0) for t in edge] + [(t * w, h) for t in edge] + [(0, t * h) for t in edge] + [(w, t * h) for t in edge]
    ) - 0.5
    world = fit.pixel_to_world(border)
    x_min, y_min = world.min(axis=0)
    x_max, y_max = world.max(axis=0)
    res = math.sqrt(fit.pixel_size[0] * fit.pixel_size[1])
    return float(x_min), float(y_max), res, int(math.ceil((x_max - x_min) / res)), int(math.ceil((y_max - y_min) / res))


def _warp(session: "MapSession", fit: "FitResult", grid: tuple[float, float, float, int, int]) -> np.ndarray:
    x_min, y_max, res, out_w, out_h = grid
    out = np.full((out_h, out_w, 3), 255, np.uint8)
    xs = x_min + (np.arange(out_w) + 0.5) * res
    for r0 in range(0, out_h, 512):
        r1 = min(out_h, r0 + 512)
        ys = y_max - (np.arange(r0, r1) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)
        src = fit.world_to_pixel(np.column_stack([gx.ravel(), gy.ravel()]))
        map_x = src[:, 0].reshape(gx.shape).astype(np.float32)
        map_y = src[:, 1].reshape(gx.shape).astype(np.float32)
        out[r0:r1] = cv2.remap(session.image, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out


def write_geotiff(session: "MapSession", fit: "FitResult", path: Path) -> dict[str, Any]:
    """GeoTIFF con el norte arriba. Si la imagen ya está alineada no se remuestrea."""
    import rasterio
    from rasterio.transform import Affine

    if _is_axis_aligned(fit):
        image = session.image
        corner = fit.pixel_to_world([[-0.5, -0.5]])[0]
        transform = Affine(fit.affine[0, 0], 0.0, corner[0], 0.0, fit.affine[1, 1], corner[1])
        resampled = False
    else:
        grid = _output_grid(session, fit)
        if max(grid[3], grid[4], session.width, session.height) > _MAX_WARP_SIDE:
            raise ValueError("La imagen es demasiado grande para remuestrearla; use el world file y los puntos de control.")
        image = _warp(session, fit, grid)
        transform = Affine(grid[2], 0.0, grid[0], 0.0, -grid[2], grid[1])
        resampled = True

    profile = {
        "driver": "GTiff", "height": image.shape[0], "width": image.shape[1], "count": 3,
        "dtype": "uint8", "transform": transform, "compress": "deflate", "tiled": True,
        "photometric": "RGB", "BIGTIFF": "IF_SAFER",
    }
    if fit.epsg:
        profile["crs"] = f"EPSG:{fit.epsg}"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
    return {"resampled": resampled, "size": [image.shape[1], image.shape[0]]}


def write_points(session: "MapSession", fit: "FitResult", path: Path) -> None:
    """Puntos de control en el formato del georreferenciador de QGIS."""
    from ..coords.crs import world_coords

    lines = ["mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"]
    for g in session.gcps:
        raw = np.array([[g.x, g.y]])
        wx, wy = (world_coords(session, fit.epsg, raw) if fit.epsg else raw)[0]
        dx, dy = fit.residual_vectors.get(g.id, (0.0, 0.0))
        # QGIS mide la fila hacia arriba: sourceY es negativa.
        lines.append(f"{wx:.6f},{wy:.6f},{g.px:.3f},{-g.py:.3f},{int(g.enabled)},{dx:.4f},{dy:.4f},{math.hypot(dx, dy):.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_world_file(session: "MapSession", fit: "FitResult", out_dir: Path) -> list[str]:
    """World file de la imagen original (aproximación afín) y su .prj."""
    a = fit.affine
    ext = _WORLD_EXT.get(session.path.suffix.lower(), ".wld")
    world = out_dir / f"{session.name}{ext}"
    values = [a[0, 0], a[0, 1], a[1, 0], a[1, 1], a[2, 0], a[2, 1]]
    world.write_text("\n".join(f"{v:.10f}" for v in values) + "\n", encoding="utf-8")
    written = [world.name]
    if fit.epsg:
        from pyproj import CRS

        prj = out_dir / f"{session.name}.prj"
        prj.write_text(CRS.from_epsg(fit.epsg).to_wkt(version="WKT1_ESRI"), encoding="utf-8")
        written.append(prj.name)
    return written


def write_overlay(session: "MapSession", path: Path, max_side: int = 2600) -> None:
    show = {"lines", "values", "labels", "gcps", "residuals"}
    canvas, _ = render_view(session, (0, 0, session.width, session.height), max_side, ref_grid=False, overlay=show)
    _write_bytes(path, ".png", canvas)


def export_all(session: "MapSession", out_dir: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    """Escribe todos los entregables de un mapa y devuelve el contenido de georef.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit = session.fit
    files: list[str] = []
    doc: dict[str, Any] = {
        "map": session.path.name,
        "image_size": [session.width, session.height],
        "dpi": session.dpi,
        **report,
        "crs": {"epsg": session.crs_epsg, "name": crs_name(session.crs_epsg),
                "confident": session.crs_confident, "evidence": session.crs_evidence},
        "label_kind": session.label_kind,
        "axis_fits": {a: f.to_dict() for a, f in session.axis_fits.items()},
        "warnings": session.warnings,
    }

    write_overlay(session, out_dir / "qa_overlay.png")
    files.append("qa_overlay.png")

    if fit is not None:
        doc["transform"] = {**fit.summary(), "affine": fit.affine.tolist()}
        doc["gcps"] = [g.to_dict() for g in session.gcps]
        write_points(session, fit, out_dir / f"{session.name}.points")
        files.append(f"{session.name}.points")
        files += write_world_file(session, fit, out_dir)
        try:
            doc["geotiff"] = write_geotiff(session, fit, out_dir / f"{session.name}_georef.tif")
            files.append(f"{session.name}_georef.tif")
        except ValueError as exc:
            session.warnings.append(str(exc))

    doc["files"] = files
    (out_dir / "georef.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc
