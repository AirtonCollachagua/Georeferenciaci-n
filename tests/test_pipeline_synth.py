"""El pipeline determinista contra mapas sintéticos con verdad conocida, sin servicios externos."""

import numpy as np
import pytest

from georef.config import Settings
from georef.eval.metrics import error_vs_truth
from georef.eval.run_eval import SYNTHETIC_SUITE, run_case
from georef.eval.synth import FakeOcr, SynthSpec, make_map
from georef.pipeline import decide_status, run_prepass
from georef.session import MapSession


@pytest.mark.parametrize("name, spec, ocr", SYNTHETIC_SUITE, ids=[c[0] for c in SYNTHETIC_SUITE])
def test_synthetic_suite(name, spec, ocr):
    row = run_case(name, spec, ocr, Settings())
    if spec.style == "none":
        assert row["status"] == "no_coordinates"
        return
    assert row["status"] == "ok"
    assert row["epsg"] == spec.epsg
    assert row["rmse_px"] < 0.5 and row["max_px"] < 1.0


def test_wrong_ocr_readings_are_rejected():
    """Etiquetas con un dígito alterado no deben mover la grilla."""
    synth = make_map(SynthSpec(rotation_deg=0.6, seed=4))
    session = MapSession(synth.image, "ruido.png")
    session.ocr_provider = FakeOcr(synth, corrupt=0.35, seed=11)
    run_prepass(session, Settings())
    assert any(lab.status == "outlier" for lab in session.labels)
    assert error_vs_truth(session, synth.pixel_to_world)["max_px"] < 0.5


def test_non_grid_strokes_get_no_value(utm_map):
    """El marco, las filas de texto y los trazos sueltos no reciben coordenadas."""
    synth, session, _ = utm_map
    valued = [ln for ln in session.lines if ln.value is not None]
    assert len(valued) < len(session.lines)
    for ln in valued:
        pos = ln.pos(session.width, session.height)
        centre = (pos, session.height / 2) if ln.axis == "v" else (session.width / 2, pos)
        true = synth.pixel_to_world([centre])[0][0 if ln.axis == "v" else 1]
        assert abs(true - ln.value) < 0.5 * synth.spec.res  # menos de medio píxel


def test_missing_zone_goes_to_review():
    """Sin zona ni datum en la leyenda hay ajuste, pero el mapa no se da por bueno."""
    synth = make_map(SynthSpec(legend="CARTA TOPOGRAFICA - ESCALA 1:50 000", seed=5))
    session = MapSession(synth.image, "sin_zona.png")
    session.ocr_provider = FakeOcr(synth)
    run_prepass(session, Settings())
    assert session.fit is not None and session.fit.rmse_px < 0.5
    assert not session.crs_confident
    assert decide_status(session, Settings())["status"] == "needs_review"


def test_region_hint_and_default_datum_resolve_the_crs():
    synth = make_map(SynthSpec(legend="CARTA TOPOGRAFICA", seed=5))
    session = MapSession(synth.image, "pista.png")
    session.ocr_provider = FakeOcr(synth)
    settings = Settings(region_hint=["18S"], default_datum="PSAD56")
    run_prepass(session, settings)
    assert session.crs_epsg == 24878
    assert decide_status(session, settings)["status"] == "ok"


def test_without_ocr_the_map_is_not_classified_as_empty():
    synth = make_map(SynthSpec(seed=6))
    session = MapSession(synth.image, "sin_ocr.png")
    run_prepass(session, Settings())
    assert len(session.lines) > 4 and not session.labels
    assert decide_status(session, Settings())["status"] == "needs_review"


def test_ground_truth_of_rotated_synthetic_map_is_consistent():
    """La verdad del generador: una etiqueta dibujada cae sobre la coordenada que dice."""
    synth = make_map(SynthSpec(rotation_deg=1.5, seed=7, label_format="plain"))
    top = [it for it in synth.labels if it.text.isdigit() and len(it.text) == 6 and it.cy < synth.spec.margin]
    for it in top:
        easting = synth.pixel_to_world([[it.cx, it.cy]])[0][0]
        assert abs(easting - float(it.text)) < 3 * synth.spec.res
    assert np.isfinite(synth.truth).all()
