"""Sistema de referencia: qué dice la leyenda y qué candidatos hay."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj import CRS, Transformer

from .parse import find_datum, find_scale, find_zone

if TYPE_CHECKING:
    from ..session import MapSession

# Código EPSG de la proyección UTM = base + zona. (base norte, base sur)
_UTM_BASE: dict[str, tuple[int | None, int | None]] = {
    "WGS84": (32600, 32700),
    "PSAD56": (24800, 24860),
    "SIRGAS2000": (31954, 31960),
    "SAD69": (29150, 29170),
    "NAD27": (26700, None),
    "NAD83": (26900, None),
    "ETRS89": (25800, None),
    "ED50": (23000, None),
}
_GEOGRAPHIC = {
    "WGS84": 4326, "PSAD56": 4248, "SIRGAS2000": 4674, "SAD69": 4618,
    "NAD27": 4267, "NAD83": 4269, "ETRS89": 4258, "ED50": 4230,
}
DATUMS = tuple(_UTM_BASE)


@lru_cache(maxsize=512)
def utm_epsg(datum: str, zone: int, south: bool) -> int | None:
    """EPSG de la zona UTM en ese datum, o None si esa combinación no existe."""
    base = _UTM_BASE.get(datum.upper(), (None, None))[1 if south else 0]
    if base is None or not 1 <= zone <= 60:
        return None
    code = base + zone
    try:
        name = CRS.from_epsg(code).name
    except Exception:
        return None
    return code if f"UTM zone {zone}{'S' if south else 'N'}" in name else None


def geographic_epsg(datum: str) -> int:
    return _GEOGRAPHIC.get(datum.upper(), 4326)


def crs_name(epsg: int | None) -> str:
    if not epsg:
        return ""
    try:
        return CRS.from_epsg(epsg).name
    except Exception:
        return f"EPSG:{epsg} (desconocido)"


def parse_region_hint(hints: list[str]) -> list[tuple[int, bool | None]]:
    """["17S", "18S", "19"] -> [(17, True), (18, True), (19, None)]"""
    out = []
    for h in hints:
        h = h.strip().upper()
        digits = "".join(ch for ch in h if ch.isdigit())
        if not digits:
            continue
        south = True if h.endswith("S") else False if h.endswith("N") else None
        out.append((int(digits), south))
    return out


def read_legend(session: "MapSession") -> dict[str, Any]:
    """Datum, zona y escala encontrados en el texto del mapa."""
    info: dict[str, Any] = {"datum": None, "zone": None, "south": None, "scale": None, "evidence": []}
    for item in session.ocr.items:
        if info["datum"] is None and (d := find_datum(item.text)):
            info["datum"] = d[0]
            info["evidence"].append(f'datum {d[0]}: "{item.text.strip()[:80]}"')
        if info["zone"] is None and (z := find_zone(item.text)):
            info["zone"], info["south"] = z[0], z[1]
            info["evidence"].append(f'zona {z[0]}: "{item.text.strip()[:80]}"')
        if info["scale"] is None and (s := find_scale(item.text)):
            info["scale"] = s
    # La zona y el hemisferio a veces van en textos distintos ("ZONA 18" ... "SUR").
    if info["zone"] is not None and info["south"] is None:
        text = session.ocr.full_text().upper()
        if "HEMISFERIO SUR" in text or "SOUTHERN HEMISPHERE" in text:
            info["south"] = True
        elif "HEMISFERIO NORTE" in text or "NORTHERN HEMISPHERE" in text:
            info["south"] = False
    return info


def _gcp_lonlat_center(session: "MapSession") -> tuple[float, float] | None:
    pts = [(g.x, g.y) for g in session.gcps if g.enabled]
    if not pts:
        return None
    arr = np.array(pts)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def propose_crs(session: "MapSession", region_hint: list[str], default_datum: str) -> list[dict[str, Any]]:
    """Candidatos de CRS con su evidencia. Fija el CRS de la sesión si la evidencia alcanza."""
    legend = read_legend(session)
    evidence = list(legend["evidence"])
    datum = legend["datum"]
    if datum is None and default_datum:
        datum = default_datum
        evidence.append(f"datum {datum}: valor por defecto configurado")
    datum_known = datum is not None
    datum = datum or "WGS84"

    candidates: list[dict[str, Any]] = []
    if session.label_kind == "geo":
        geo = geographic_epsg(datum)
        candidates.append({"epsg": geo, "name": crs_name(geo), "role": "geográfico"})
        center = _gcp_lonlat_center(session)
        if center:
            lon, lat = center
            zone = int((lon + 180) // 6) + 1
            code = utm_epsg(datum, zone, lat < 0)
            if code:
                candidates.append({"epsg": code, "name": crs_name(code), "role": "proyectado (UTM por longitud)"})
        confident = datum_known
        if not datum_known:
            evidence.append("datum no encontrado en la leyenda: se asume WGS84")
    else:
        zones: list[tuple[int, bool | None]] = []
        zone_from_legend = legend["zone"] is not None
        if zone_from_legend:
            zones = [(legend["zone"], legend["south"])]
        else:
            zones = parse_region_hint(region_hint)
            if zones:
                evidence.append("zona: tomada de la pista regional configurada")
        north_values = [g.y for g in session.gcps if g.enabled]
        for zone, south in zones:
            options = [south] if south is not None else [True, False]
            for s in options:
                # Ningún Norte del hemisferio norte pasa de ~9 330 000 m.
                if s is False and north_values and max(north_values) > 9_400_000:
                    continue
                code = utm_epsg(datum, zone, s)
                if code:
                    candidates.append({"epsg": code, "name": crs_name(code), "role": "proyectado"})
        hemisphere_known = bool(zones) and all(s is not None for _, s in zones)
        confident = datum_known and len(candidates) == 1 and (zone_from_legend or len(zones) == 1) and hemisphere_known
        if not zones:
            evidence.append("zona UTM no encontrada: los valores Este se repiten en todas las zonas")
        if not datum_known:
            evidence.append("datum no encontrado en la leyenda: se asume WGS84")

    session.crs_evidence = evidence
    session.crs_confident = bool(confident and candidates)
    if candidates and (session.crs_confident or len(candidates) == 1 or session.label_kind == "geo"):
        if session.crs_epsg not in {c["epsg"] for c in candidates}:
            session.crs_epsg = candidates[0]["epsg"]
    return candidates


def world_coords(session: "MapSession", epsg: int, points: np.ndarray) -> np.ndarray:
    """Lleva coordenadas en unidades de las etiquetas al CRS `epsg`."""
    crs = CRS.from_epsg(epsg)
    if session.label_kind != "geo" or crs.is_geographic:
        return points.astype(float)
    tr = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    x, y = tr.transform(points[:, 0], points[:, 1])
    return np.column_stack([x, y])


def area_of_use_ok(epsg: int, x: float, y: float) -> tuple[bool, str]:
    """¿Cae el punto (en el CRS dado) dentro del área de uso del CRS?"""
    crs = CRS.from_epsg(epsg)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    area = crs.area_of_use
    if area is None or not np.isfinite([lon, lat]).all():
        return False, "no se pudo evaluar"
    inside = area.west - 1 <= lon <= area.east + 1 and area.south - 1 <= lat <= area.north + 1
    return inside, f"centro en lon {lon:.3f}, lat {lat:.3f}; área de uso: {area.name}"
