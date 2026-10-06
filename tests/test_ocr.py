"""OCR: agrupación de palabras, geometría de los recortes girados e índice."""

import numpy as np
import pytest
from google.cloud import documentai

from georef.ocr.docai import group_tokens, parse_document
from georef.ocr.index import OcrIndex
from georef.ocr.tiling import iter_tiles, run_ocr
from georef.session import MapSession, OcrItem


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def test_words_of_one_label_are_joined_and_distant_labels_are_not():
    tokens = [
        ("8", box(100, 50, 112, 70), 0.9), ("650", box(120, 50, 156, 70), 0.9), ("000", box(164, 50, 200, 70), 0.9),
        ("8", box(900, 50, 912, 70), 0.9), ("651", box(920, 50, 956, 70), 0.9), ("000", box(964, 50, 1000, 70), 0.9),
        ("N", box(203, 50, 215, 70), 0.9),
    ]
    phrases = sorted(group_tokens(tokens), key=lambda p: p.x0)
    assert [p.text for p in phrases] == ["8 650 000 N", "8 651 000"]
    assert (phrases[0].x0, phrases[0].x1) == (100, 215)


def test_vertical_text_is_grouped_along_its_reading_direction():
    """Texto que se lee de abajo hacia arriba: el polígono llega en orden de lectura."""
    def upward(x0, y0, x1, y1):  # primer vértice: abajo a la izquierda; el segundo: arriba a la izquierda
        return np.array([[x0, y1], [x0, y0], [x1, y0], [x1, y1]], dtype=float)

    tokens = [("8", upward(40, 480, 60, 492), 0.9), ("650", upward(40, 436, 60, 472), 0.9), ("000", upward(40, 392, 60, 428), 0.9)]
    (phrase,) = group_tokens(tokens)
    assert phrase.text == "8 650 000" and phrase.rotation == 270
    assert (phrase.x0, phrase.y0, phrase.x1, phrase.y1) == (40, 392, 60, 492)


def test_document_ai_response_is_parsed_with_its_real_classes():
    """La respuesta se arma con las clases de la librería: detecta cambios de nombre de campos."""
    text = "320 000 E\n"

    def token(start, end, x0, x1):
        verts = [documentai.NormalizedVertex(x=x, y=y) for x, y in ((x0, 0.1), (x1, 0.1), (x1, 0.2), (x0, 0.2))]
        return documentai.Document.Page.Token(layout=documentai.Document.Page.Layout(
            text_anchor=documentai.Document.TextAnchor(
                text_segments=[documentai.Document.TextAnchor.TextSegment(start_index=start, end_index=end)]),
            bounding_poly=documentai.BoundingPoly(normalized_vertices=verts), confidence=0.98))

    document = documentai.Document(text=text, pages=[documentai.Document.Page(
        tokens=[token(0, 3, 0.10, 0.16), token(4, 7, 0.17, 0.23), token(8, 9, 0.24, 0.26)])])
    (item,) = parse_document(document, width=1000, height=500)
    assert item.text == "320 000 E" and item.conf == pytest.approx(0.98)
    assert (item.x0, item.y0, item.x1, item.y1) == pytest.approx((100, 50, 260, 100))

    request = documentai.ProcessRequest(name="projects/p/locations/us/processors/x",
                                        raw_document=documentai.RawDocument(content=b"png", mime_type="image/png"))
    assert request.raw_document.mime_type == "image/png"


class BlobFinder:
    """Proveedor de prueba: "lee" la mancha oscura del recorte que recibe."""

    def recognize(self, image):
        ys, xs = np.nonzero(image[:, :, 0] < 100)
        return [OcrItem("X", float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 0.9)]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("upscale", [1.0, 2.0])
def test_boxes_from_rotated_and_enlarged_crops_map_back_to_the_map(rotation, upscale):
    image = np.full((400, 600, 3), 255, np.uint8)
    image[150:170, 320:400] = 0  # mancha en x 320..399, y 150..169
    session = MapSession(image, "blob.png")
    session.ocr_provider = BlobFinder()
    (item,) = run_ocr(session, (250, 100, 300, 200), rotation=rotation, upscale=upscale)
    assert (item.x0, item.y0, item.x1, item.y1) == pytest.approx((320, 150, 399, 169), abs=1.0)


def test_tiles_cover_the_region_with_overlap():
    tiles = iter_tiles(0, 0, 5000, 3000, tile=2200, overlap=260)
    covered = np.zeros((3000, 5000), bool)
    for x, y, w, h in tiles:
        assert w <= 2200 and h <= 2200
        covered[y : y + h, x : x + w] = True
    assert covered.all() and len(tiles) == 6
    assert iter_tiles(10, 20, 500, 400) == [(10, 20, 500, 400)]


def test_index_merges_repeated_readings_and_searches():
    index = OcrIndex()
    assert index.add([OcrItem("320 0O0", 100, 50, 200, 70, 0.6), OcrItem("DATUM WGS 84", 300, 900, 520, 925, 0.9)]) == 2
    # El mismo texto visto en el mosaico vecino, mejor leído: reemplaza al anterior.
    assert index.add([OcrItem("320 000", 101, 51, 201, 71, 0.95), OcrItem("  ", 0, 0, 5, 5, 0.9)]) == 0
    assert [it.text for it in index.search(r"\d{3} \d{3}")] == ["320 000"]
    assert [it.text for it in index.search(region=(250, 850, 400, 200))] == ["DATUM WGS 84"]
    with pytest.raises(ValueError):
        index.search("(")
