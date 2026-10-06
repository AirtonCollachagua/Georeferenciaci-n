"""Las herramientas del agente, llamadas directamente y sin modelo."""

import base64
import json

import cv2
import numpy as np
import pytest

from georef.agent.tools import build_tools, parse_value
from georef.config import Settings
from georef.eval.metrics import error_vs_truth
from georef.eval.synth import FakeOcr, SynthSpec, make_map
from georef.grid import detect_grid
from georef.session import MapSession


@pytest.fixture
def tools(utm_map):
    synth, session, _ = utm_map
    by_name = {t.name: t for t in build_tools(session, Settings())}

    def call(name, **kwargs):
        result = by_name[name].call(kwargs)
        if isinstance(result, str):
            return json.loads(result), None
        text = json.loads(next(b["text"] for b in result if b["type"] == "text"))
        image = next((b for b in result if b["type"] == "image"), None)
        return text, image

    return synth, session, call


def _decode(block) -> np.ndarray:
    raw = base64.standard_b64decode(block["source"]["data"])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_view_image_returns_an_image_within_the_model_limit(tools):
    _, session, call = tools
    meta, image = call("view_image", max_side=4000)
    assert image["source"]["media_type"] in ("image/jpeg", "image/png")
    assert max(_decode(image).shape[:2]) <= 2576 + 30
    assert meta["region"] == {"x": 0, "y": 0, "w": session.width, "h": session.height}


def test_zoom_by_cells_and_by_pixels(tools):
    _, session, call = tools
    by_cell, _ = call("view_image", region="B2:C3")
    cell = session.ref_grid.cell
    assert by_cell["region"] == {"x": cell, "y": cell, "w": 2 * cell, "h": 2 * cell}
    by_pixel, image = call("view_image", region="150,150,200,120", max_side=1024)
    assert by_pixel["scale"] == pytest.approx(4.0)  # un recorte pequeño se amplía
    assert image is not None


def test_bad_arguments_come_back_as_an_error_the_model_can_read(tools):
    _, _, call = tools
    for kwargs in ({"region": "Z99"}, {"region": "1,2,3"}, {"enhance": "sepia"}):
        result, image = call("view_image", **kwargs)
        assert "error" in result and image is None


def test_view_grid_lists_every_line_with_its_value(tools):
    _, session, call = tools
    meta, image = call("view_grid")
    assert image is not None
    assert len(meta["lines"]["V"]) == len(session.lines_of("v"))
    assert any("=320000" in tag for tag in meta["lines"]["V"])


def test_search_text_finds_the_legend(tools):
    _, session, call = tools
    assert call("search_text", kind="datum")[0]["total"] == 1
    assert call("search_text", kind="zone")[0]["total"] == 1
    assert call("search_text", pattern="chillon")[0]["matches"][0]["text"] == "RIO CHILLON"
    top_row = f"A1:{session.ref_grid.name(session.ref_grid.cols - 1, 0)}"
    assert call("search_text", kind="utm", region=top_row)[0]["total"] > 0
    assert call("search_text", kind="geo")[0]["total"] == 0


def test_ocr_region_adds_text_to_the_index(tools):
    synth, session, call = tools
    session.ocr.items.clear()
    result, _ = call("ocr_region", region=f"0,0,{session.width},{synth.spec.margin}")
    assert result["new_in_index"] == len(result["texts"]) > 5
    assert any(t.get("coordinate") for t in result["texts"])


def test_manual_values_override_the_ocr_and_propagate(tools):
    """Dos líneas fijadas por el agente redefinen el eje completo."""
    _, session, call = tools
    valued = [ln for ln in session.lines_of("v") if ln.value is not None]
    first, second = valued[0], valued[1]
    result, _ = call("assign_labels", assignments=[
        {"line_id": first.id, "value": str(int(first.value) + 5000)},
        {"line_id": second.id, "value": str(int(second.value) + 5000)},
    ])
    assert "error" not in result
    shifted = [ln for ln in session.lines_of("v") if ln.value is not None]
    assert len(shifted) == len(valued)
    assert shifted[-1].value == valued[-1].value  # mismo objeto: ya trae el nuevo valor
    assert shifted[2].value - shifted[1].value == 1000
    assert result["outlier_labels"]  # las etiquetas del OCR ya no encajan en ese eje


def test_a_line_can_be_excluded(tools):
    _, session, call = tools
    target = next(ln for ln in session.lines_of("h") if ln.value is not None)
    before = sum(g.enabled for g in session.gcps)
    call("assign_labels", assignments=[{"line_id": target.id, "value": "none"}])
    assert session.line(target.id).value is None
    assert sum(g.enabled for g in session.gcps) < before


def test_edit_gcps_disable_line_and_add_snapped_point(tools):
    synth, session, call = tools
    listing, _ = call("edit_gcps", action="list")
    assert listing["total"] == listing["enabled"] == len(session.gcps)

    line = next(ln for ln in session.lines_of("v") if ln.value is not None)
    changed, _ = call("edit_gcps", action="disable", gcp_id=line.id)
    assert changed["changed"] and all(c.startswith(line.id + "H") for c in changed["changed"])
    assert changed["fit"]["n_gcps"] == listing["enabled"] - len(changed["changed"])

    target = [g for g in session.gcps if g.enabled][-1]  # lejos del trazo señuelo del mapa sintético
    added, _ = call("edit_gcps", action="add", px=target.px + 4, py=target.py - 3, x=str(target.x), y=str(target.y))
    assert np.hypot(added["pixel"][0] - target.px, added["pixel"][1] - target.py) < 1.0
    assert error_vs_truth(session, synth.pixel_to_world)["max_px"] < 0.5


def test_crs_tools(tools):
    _, session, call = tools
    ranking, _ = call("rank_crs")
    assert ranking["legend"]["datum"] == "WGS84" and ranking["legend"]["zone"] == 18
    assert "error" in call("set_crs", epsg=24878, evidence="  ")[0]
    assert "error" in call("set_crs", epsg=999999, evidence="x")[0]
    result, _ = call("set_crs", epsg=24878, evidence='leyenda: "PSAD 56"')
    assert session.crs_epsg == 24878 and result["fit"]["epsg"] == 24878


def test_submit_result_closes_the_session(tools):
    _, session, call = tools
    assert "error" in call("submit_result", status="ok", epsg=4326, confidence=0.9, notes="x")[0]
    assert session.result is None
    call("submit_result", status="ok", epsg=32718, confidence=1.7, notes=" Comprobado. ")
    assert session.result == {"status": "ok", "epsg": 32718, "confidence": 1.0, "notes": "Comprobado."}


def test_detect_grid_can_be_rerun_in_another_mode():
    synth = make_map(SynthSpec(style="crosses", rotation_deg=0.9, seed=9))
    session = MapSession(synth.image, "cruces.png")
    session.ocr_provider = FakeOcr(synth)
    detect_grid(session, "lines")
    call = {t.name: t for t in build_tools(session, Settings())}
    session.ocr.add(session.ocr_provider.recognize_region((0, 0, session.width, session.height)))
    result = json.loads(call["detect_grid"].call({"mode": "crosses"}))
    assert result["info"]["mode"] == "crosses" and result["fit"]["rmse_px"] < 0.5
    assert error_vs_truth(session, synth.pixel_to_world)["max_px"] < 0.5


@pytest.mark.parametrize("text, value", [("320000", 320000.0), ("-76.75", -76.75), ("76°30' W", -76.5), ("8 650 000 N", 8650000.0)])
def test_parse_value(text, value):
    assert parse_value(text) == pytest.approx(value)


def test_parse_value_rejects_garbage():
    with pytest.raises(ValueError):
        parse_value("no es un número")
