"""Interpretación de texto como coordenadas y como datos de la leyenda."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Candidate:
    kind: str  # utm | geo
    value: float | None  # None si la etiqueta está abreviada
    axis: str | None  # x (Este / longitud) | y (Norte / latitud)
    complete: bool
    signed: bool = False  # geo: el texto traía hemisferio o signo
    digits: str = ""


def normalize(text: str) -> str:
    """Unifica símbolos de grado, minuto y segundo, y deshace los superíndices."""
    # Antes de NFKC: esa normalización convierte "º" en la letra "o".
    for ch in "º˚":
        text = text.replace(ch, "°")
    t = unicodedata.normalize("NFKC", text)
    for ch in "’‘′´`":
        t = t.replace(ch, "'")
    for ch in "″”“":
        t = t.replace(ch, '"')
    return t.replace("''", '"')


_HEMISPHERE_WORDS = [
    (r"\b(OESTE|WEST)\b", "W"),
    (r"\b(ESTE|EAST)\b", "E"),
    (r"\b(NORTE|NORTH)\b", "N"),
    (r"\b(SUR|SOUTH)\b", "S"),
]

_DMS = re.compile(
    r"""
    (?:(?<![A-Za-z])(?P<pre>[NSEWO])\s*)?
    (?P<sign>-)?\s*
    (?P<d>\d{1,3}(?:[.,]\d+)?)\s*°\s*
    (?:(?P<m>\d{1,2}(?:[.,]\d+)?)\s*'\s*
       (?:(?P<s>\d{1,2}(?:[.,]\d+)?)\s*"?\s*)?
    )?
    (?:(?P<post>[NSEWO])(?![A-Za-z]))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:[ .,']\d{3})+|\d{2,8})(?!\d)")
_LABEL_ONLY = re.compile(r"^[\d\s.,']+\s*(?:m\.?)?\s*[EN]?\.?$", re.IGNORECASE)
_AXIS_AFTER = re.compile(r"^\s*(?:m\.?)?\s*([EN])(?![A-Za-z])", re.IGNORECASE)
_AXIS_BEFORE = re.compile(r"(?<![A-Za-z])([EN])\s*[:=.]?\s*$", re.IGNORECASE)
_NOT_COORD_AFTER = re.compile(r"^\s*(km|ha|has|msnm|m\.s\.n\.m|mm|cm|%)\b", re.IGNORECASE)
_SCALE_BEFORE = re.compile(r"1\s*[:/]\s*$")


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def _parse_geo(text: str) -> tuple[list[Candidate], str]:
    """Devuelve las coordenadas geográficas y el texto con esos tramos tachados."""
    found = []
    masked = text
    for m in _DMS.finditer(text):
        deg = _num(m["d"])
        minutes = _num(m["m"]) if m["m"] else 0.0
        seconds = _num(m["s"]) if m["s"] else 0.0
        if deg > 180 or minutes >= 60 or seconds >= 60:
            continue
        hemisphere = (m["post"] or m["pre"] or "").upper()
        value = deg + minutes / 60 + seconds / 3600
        if m["sign"] or hemisphere in {"S", "W", "O"}:
            value = -value
        axis = "y" if hemisphere in {"N", "S"} else "x" if hemisphere in {"E", "W", "O"} else None
        if axis is None and deg > 90:
            axis = "x"
        found.append(Candidate("geo", value, axis, True, signed=bool(m["sign"] or hemisphere)))
        masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]
    return found, masked


def _utm_candidate(digits: str, axis: str | None, label_like: bool) -> Candidate | None:
    n = int(digits)
    if 100_000 <= n <= 999_999:
        return Candidate("utm", float(n), axis, True, digits=digits)
    if 1_000_000 <= n <= 10_000_000:
        return Candidate("utm", float(n), axis or "y", True, digits=digits)
    # Abreviada ("51" por 8 651 000): solo si el texto es la etiqueta y nada más.
    if label_like and 2 <= len(digits) <= 5:
        return Candidate("utm", None, axis, False, digits=digits)
    return None


def _parse_utm(text: str) -> list[Candidate]:
    stripped = text.strip()
    if _LABEL_ONLY.match(stripped):
        digits = re.sub(r"\D", "", stripped)
        letter = re.search(r"([EN])\.?$", stripped, re.IGNORECASE)
        axis = {"E": "x", "N": "y"}.get(letter.group(1).upper()) if letter else None
        if digits:
            cand = _utm_candidate(digits, axis, label_like=True)
            return [cand] if cand else []

    found = []
    for m in _NUMBER.finditer(text):
        before, after = text[: m.start()], text[m.end() :]
        if _SCALE_BEFORE.search(before) or _NOT_COORD_AFTER.match(after):
            continue
        digits = re.sub(r"\D", "", m.group(1))
        hint = _AXIS_AFTER.match(after) or _AXIS_BEFORE.search(before)
        axis = {"E": "x", "N": "y"}.get(hint.group(1).upper()) if hint else None
        cand = _utm_candidate(digits, axis, label_like=False)
        if cand:
            found.append(cand)
    return found


def parse_candidates(text: str) -> list[Candidate]:
    """Coordenadas presentes en un texto reconocido por OCR."""
    t = normalize(text)
    for pattern, letter in _HEMISPHERE_WORDS:
        t = re.sub(pattern, letter, t, flags=re.IGNORECASE)
    geo, rest = _parse_geo(t)
    return geo + _parse_utm(rest)


def format_dms(value: float) -> str:
    sign = "-" if value < 0 else ""
    total = round(abs(value) * 3600)
    d, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{d}°{m:02d}'{s:02d}\""


def format_value(kind: str, value: float) -> str:
    return format_dms(value) if kind == "geo" else f"{value:.0f}"


# --- Datos de la leyenda ---

_DATUMS = [
    ("WGS84", r"W\.?\s*G\.?\s*S\.?\s*[- ]?\s*(19)?84"),
    ("PSAD56", r"P\.?\s*S\.?\s*A\.?\s*D\.?\s*[- ]?\s*(19)?56|PROVISIONAL\s+(DE\s+)?(SOUTH|SUD)\s*AM[EÉ]RICA\w*\s*(DE\s+)?(19)?56|PROVISIONAL\s+SUDAMERICANO"),
    ("SIRGAS2000", r"SIRGAS"),
    ("SAD69", r"S\.?\s*A\.?\s*D\.?\s*[- ]?\s*(19)?69"),
    ("NAD27", r"NAD\s*[- ]?\s*(19)?27"),
    ("NAD83", r"NAD\s*[- ]?\s*(19)?83"),
    ("ETRS89", r"ETRS\s*[- ]?\s*89"),
    ("ED50", r"\bED\s*[- ]?\s*50\b|EUROPEAN\s+DATUM\s+1950"),
]

_ZONE = re.compile(
    r"(?:ZONA|ZONE|HUSO|FUSO)\s*(?:UTM\s*)?(?:N[°ºo.]?\s*)?(\d{1,2})\s*([A-Z])?(?![A-Za-z])"
    r"|UTM\s*(?:ZONA|ZONE)?\s*(\d{1,2})\s*([A-Z])(?![A-Za-z])",
    re.IGNORECASE,
)
_SCALE = re.compile(r"1\s*[:/]\s*(\d{1,3}(?:[ .,']\d{3})+|\d{3,8})")


def find_datum(text: str) -> tuple[str, str] | None:
    """(datum, texto encontrado) según la leyenda."""
    t = normalize(text)
    for name, pattern in _DATUMS:
        m = re.search(pattern, t, re.IGNORECASE)
        if m:
            return name, m.group(0)
    return None


def find_zone(text: str) -> tuple[int, bool | None, str] | None:
    """(zona, es_sur, texto encontrado). `es_sur` es None si el texto no lo dice."""
    t = normalize(text)
    m = _ZONE.search(t)
    if not m:
        return None
    zone = int(m.group(1) or m.group(3))
    if not 1 <= zone <= 60:
        return None
    letter = (m.group(2) or m.group(4) or "").upper()
    tail = t[m.end() : m.end() + 14].upper()
    south: bool | None = None
    if re.match(r"\s*(SUR|SOUTH)\b", tail) or letter == "S":
        south = True  # en la cartografía local "18S" es zona 18 Sur, no la banda S
    elif re.match(r"\s*(NORTE|NORTH)\b", tail) or letter == "N":
        south = False
    elif letter and letter in "CDEFGHJKLM":
        south = True  # bandas de latitud MGRS del hemisferio sur
    elif letter and letter in "PQRTUVWX":
        south = False
    return zone, south, m.group(0).strip()


def find_scale(text: str) -> int | None:
    m = _SCALE.search(normalize(text))
    if not m:
        return None
    value = int(re.sub(r"\D", "", m.group(1)))
    return value if 500 <= value <= 5_000_000 else None
