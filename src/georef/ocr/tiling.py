"""OCR por recortes y por mosaicos, con el resultado en píxeles de la imagen original."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..session import OcrItem

if TYPE_CHECKING:
    from ..session import MapSession

ROTATIONS = (0, 90, 180, 270)
# Giros de np.rot90 (antihorarios) que equivalen a girar la imagen `rotation` grados en sentido horario.
_ROT90_K = {0: 0, 90: 3, 180: 2, 270: 1}


def iter_tiles(x: int, y: int, w: int, h: int, tile: int = 2200, overlap: int = 260) -> list[tuple[int, int, int, int]]:
    """Mosaicos solapados que cubren la región."""
    step = tile - overlap
    tiles = []
    ty = y
    while True:
        th = min(tile, y + h - ty)
        tx = x
        while True:
            tw = min(tile, x + w - tx)
            tiles.append((tx, ty, tw, th))
            if tx + tw >= x + w:
                break
            tx += step
        if ty + th >= y + h:
            break
        ty += step
    return tiles


def _unrotate(xr: float, yr: float, rotation: int, wc: int, hc: int) -> tuple[float, float]:
    """Punto del recorte girado -> punto del recorte sin girar (de tamaño wc x hc)."""
    if rotation == 0:
        return xr, yr
    if rotation == 90:
        return yr, hc - 1 - xr
    if rotation == 180:
        return wc - 1 - xr, hc - 1 - yr
    return wc - 1 - yr, xr


def run_ocr(
    session: "MapSession",
    region: tuple[int, int, int, int],
    rotation: int = 0,
    upscale: float = 1.0,
) -> list[OcrItem]:
    """OCR de una región. `rotation` gira el recorte en sentido horario antes de leerlo:
    90 pone derecho el texto que se lee de abajo hacia arriba."""
    provider = session.ocr_provider
    if provider is None:
        raise RuntimeError("No hay proveedor de OCR configurado (falta DOCAI_PROCESSOR_ID).")
    if rotation not in ROTATIONS:
        raise ValueError("La rotación debe ser 0, 90, 180 o 270.")
    x, y, w, h = region

    # Un proveedor que conoce el mapa completo (pruebas) responde por región.
    if hasattr(provider, "recognize_region"):
        return provider.recognize_region(region, rotation)

    crop = session.image[y : y + h, x : x + w]
    upscale = float(min(4.0, max(0.25, upscale)))
    if upscale != 1.0:
        interp = cv2.INTER_CUBIC if upscale > 1 else cv2.INTER_AREA
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=interp)
    hc, wc = crop.shape[:2]
    rotated = np.ascontiguousarray(np.rot90(crop, _ROT90_K[rotation]))

    items = []
    for it in provider.recognize(rotated):
        corners = [
            _unrotate(px, py, rotation, wc, hc)
            for px, py in ((it.x0, it.y0), (it.x1, it.y0), (it.x1, it.y1), (it.x0, it.y1))
        ]
        xs = [c[0] / upscale + x for c in corners]
        ys = [c[1] / upscale + y for c in corners]
        items.append(
            OcrItem(it.text, min(xs), min(ys), max(xs), max(ys), it.conf, (it.rotation + rotation) % 360)
        )
    return items


def ocr_full(session: "MapSession", tile: int = 2200, overlap: int = 260, workers: int = 4) -> int:
    """OCR de toda la imagen por mosaicos. Devuelve cuántos textos nuevos entraron al índice."""
    tiles = iter_tiles(0, 0, session.width, session.height, tile, overlap)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda t: run_ocr(session, t), tiles))
    return sum(session.ocr.add(items) for items in results)


def ocr_side_bands(session: "MapSession", workers: int = 4) -> int:
    """Segunda pasada sobre los márgenes laterales con el recorte girado.

    Las coordenadas Norte suelen ir rotuladas en vertical; girar el recorte
    las deja derechas para el OCR.
    """
    frame = session.frame
    pad = int(0.02 * session.width)
    if frame is not None:
        left_w = int(min(session.width, frame.x0 + pad))
        right_x = int(max(0, frame.x1 - pad))
    else:
        left_w = int(0.12 * session.width)
        right_x = int(0.88 * session.width)
    bands = []
    if left_w > 20:
        bands.append((0, 0, left_w, session.height))
    if session.width - right_x > 20:
        bands.append((right_x, 0, session.width - right_x, session.height))

    jobs = [(t, 90) for band in bands for t in iter_tiles(*band)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: run_ocr(session, job[0], rotation=job[1]), jobs))
    return sum(session.ocr.add(items) for items in results)
