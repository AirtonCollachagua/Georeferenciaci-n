import pytest

from georef.coords.crs import parse_region_hint, utm_epsg
from georef.coords.parse import find_datum, find_scale, find_zone, format_dms, parse_candidates


def one(text):
    found = parse_candidates(text)
    assert len(found) == 1, f"{text!r} -> {found}"
    return found[0]


@pytest.mark.parametrize("text, value, axis", [
    ("8 650 000 m N", 8_650_000, "y"),
    ("8650000", 8_650_000, "y"),
    ("320 000 mE", 320_000, "x"),
    ("320000", 320_000, None),
    ("E 320000", 320_000, "x"),
    ("⁸⁶50⁰⁰⁰ᵐN", 8_650_000, "y"),  # dígitos en superíndice
    ("8'650,000 N", 8_650_000, "y"),
])
def test_utm_labels(text, value, axis):
    c = one(text)
    assert (c.kind, c.value, c.axis, c.complete) == ("utm", value, axis, True)


def test_abbreviated_utm_label_is_partial():
    c = one("51")
    assert c.kind == "utm" and c.value is None and not c.complete and c.digits == "51"


@pytest.mark.parametrize("text, value, axis, signed", [
    ("76°30'", 76.5, None, False),
    ("76°30'00\" W", -76.5, "x", True),
    ("12º15′S", -12.25, "y", True),
    ("LAT. 11°55' SUR", -(11 + 55 / 60), "y", True),
    ("77° 02' 30\" O", -(77 + 2 / 60 + 30 / 3600), "x", True),  # O de Oeste
    ("-76.75°", -76.75, None, True),
    ("120°15' E", 120.25, "x", True),
])
def test_geographic_labels(text, value, axis, signed):
    c = one(text)
    assert c.kind == "geo" and c.value == pytest.approx(value, abs=1e-9)
    assert (c.axis, c.signed) == (axis, signed)


@pytest.mark.parametrize("text", ["ESCALA 1:50 000", "1/25000", "Altitud 3450 msnm", "RIO CHILLON", "12 km"])
def test_text_that_is_not_a_coordinate(text):
    assert [c for c in parse_candidates(text) if c.complete] == []


@pytest.mark.parametrize("text, datum", [
    ("DATUM: WGS-84", "WGS84"),
    ("Datum horizontal W.G.S. 1984", "WGS84"),
    ("Datum Provisional Sudamericano 1956", "PSAD56"),
    ("PSAD 56", "PSAD56"),
    ("SIRGAS 2000", "SIRGAS2000"),
    ("SAD-69", "SAD69"),
])
def test_datum(text, datum):
    assert find_datum(text)[0] == datum


@pytest.mark.parametrize("text, zone, south", [
    ("PROYECCIÓN UTM ZONA 18 SUR", 18, True),
    ("UTM 17S", 17, True),
    ("Zone 19L", 19, True),  # banda de latitud MGRS del hemisferio sur
    ("ZONA 18", 18, None),
    ("Huso 30 Norte", 30, False),
])
def test_zone(text, zone, south):
    assert find_zone(text)[:2] == (zone, south)


def test_scale_and_dms_format():
    assert find_scale("ESCALA 1:100 000") == 100_000
    assert find_scale("sin escala") is None
    assert format_dms(-76.5) == "-76°30'00\""
    assert format_dms(11 + 55 / 60) == "11°55'00\""


@pytest.mark.parametrize("datum, zone, south, epsg", [
    ("WGS84", 18, True, 32718),
    ("WGS84", 18, False, 32618),
    ("PSAD56", 18, True, 24878),
    ("PSAD56", 17, False, 24817),
    ("SIRGAS2000", 18, True, 31978),
    ("SAD69", 18, True, 29188),
    ("NAD27", 18, True, None),  # no existe esa combinación
    ("WGS84", 61, True, None),
])
def test_utm_epsg(datum, zone, south, epsg):
    assert utm_epsg(datum, zone, south) == epsg


def test_region_hint():
    assert parse_region_hint(["17S", "18s", "19"]) == [(17, True), (18, True), (19, None)]
