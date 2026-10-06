"""Entregables: lo que se escribe a disco georreferencia de verdad."""

import json

import numpy as np
import pytest
import rasterio

from georef.config import Settings
from georef.eval.run_eval import read_points
from georef.eval.synth import FakeOcr, SynthSpec, make_map
from georef.georef.export import export_all
from georef.georef.transform import fit_transform
from georef.pipeline import decide_status, process_session
from georef.session import MapSession


def _ink_at(dataset, x, y) -> float:
    """Gris medio alrededor de una coordenada: bajo si ahí hay tinta."""
    row, col = dataset.index(x, y)
    window = dataset.read(1, window=((row - 1, row + 2), (col - 1, col + 2)))
    return float(window.mean())


def test_all_deliverables_are_written(utm_map, tmp_path):
    _, session, _ = utm_map
    doc = export_all(session, tmp_path, decide_status(session, Settings()))
    for name in ("georef.json", "qa_overlay.png", "utm_demo.points", "utm_demo.pgw", "utm_demo.prj", "utm_demo_georef.tif"):
        assert (tmp_path / name).exists(), name
    saved = json.loads((tmp_path / "georef.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ok" and saved["crs"]["epsg"] == 32718
    assert saved["transform"]["n_gcps"] == len(saved["gcps"]) == doc["transform"]["n_gcps"]


def test_geotiff_of_rotated_scan_is_north_up_and_lands_on_the_grid(utm_map, tmp_path):
    """En el GeoTIFF, cada cruce de la cuadrícula cae en la coordenada que le corresponde."""
    _, session, _ = utm_map
    doc = export_all(session, tmp_path, {})
    assert doc["geotiff"]["resampled"]
    with rasterio.open(tmp_path / "utm_demo_georef.tif") as ds:
        assert ds.crs.to_epsg() == 32718
        assert ds.transform.b == 0 and ds.transform.d == 0 and ds.transform.e < 0  # norte arriba
        darkness = [_ink_at(ds, g.x, g.y) for g in session.gcps[:40]]
        paper = float(np.median(ds.read(1)))
    assert np.median(darkness) < paper - 60


def test_axis_aligned_scan_is_written_without_resampling(tmp_path):
    synth = make_map(SynthSpec(seed=2))
    session = MapSession(synth.image, "recto.png")
    session.ocr_provider = FakeOcr(synth)
    doc = process_session(session, tmp_path, Settings(), use_agent=False)
    assert doc["status"] == "ok" and not doc["geotiff"]["resampled"]
    with rasterio.open(tmp_path / "recto_georef.tif") as ds:
        assert (ds.width, ds.height) == (session.width, session.height)
        # Centro del píxel (col, fila) -> mundo, frente a la verdad del mapa.
        for col, row in ((0, 0), (1200, 900), (2399, 1799)):
            x, y = ds.xy(row, col)  # centro del píxel
            assert np.allclose((x, y), synth.pixel_to_world([[col, row]])[0], atol=0.05)


def test_points_and_world_file_match_the_fit(utm_map, tmp_path):
    synth, session, _ = utm_map
    export_all(session, tmp_path, {})
    px, world, _ = read_points(tmp_path / "utm_demo.points")
    assert len(px) == len(session.gcps)
    assert np.allclose(world, synth.pixel_to_world(px), atol=0.5)

    a, d, b, e, c, f = (float(v) for v in (tmp_path / "utm_demo.pgw").read_text().split())
    predicted = np.array([a * 1000 + b * 500 + c, d * 1000 + e * 500 + f])
    assert np.allclose(predicted, synth.pixel_to_world([[1000, 500]])[0], atol=0.5)
    assert "UTM" in (tmp_path / "utm_demo.prj").read_text()


def test_fit_needs_three_points(utm_map):
    _, session, _ = utm_map
    for g in session.gcps[2:]:
        g.enabled = False
    with pytest.raises(ValueError, match="al menos 3"):
        fit_transform(session)


@pytest.mark.parametrize("kind", ["affine", "projective", "poly2", "tps"])
def test_every_model_recovers_a_clean_grid(utm_map, kind):
    synth, session, _ = utm_map
    fit = fit_transform(session, kind)
    assert fit.kind == kind
    probe = np.array([[300.0, 400.0], [1900.0, 1300.0]])
    assert np.allclose(fit.pixel_to_world(probe), synth.pixel_to_world(probe), atol=1.0)
    assert np.allclose(fit.world_to_pixel(fit.pixel_to_world(probe)), probe, atol=0.05)
