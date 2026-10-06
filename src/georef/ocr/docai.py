"""OCR con Google Document AI (procesador Enterprise Document OCR)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ..session import OcrItem
from .cache import OcrCache

# Document AI en línea: 40 MB por archivo y 40 megapíxeles por imagen.
MAX_BYTES = 38 * 1024 * 1024
MAX_PIXELS = 40_000_000


def group_tokens(tokens: list[tuple[str, np.ndarray, float]]) -> list[OcrItem]:
    """Une palabras contiguas en frases ("8", "650", "000" -> "8 650 000").

    Cada token trae su polígono en orden de lectura, lo que permite agrupar
    también el texto vertical de los márgenes.
    """
    by_orientation: dict[int, list[tuple[str, np.ndarray, float, tuple[float, float, float, float]]]] = {}
    for text, poly, conf in tokens:
        if not text.strip() or len(poly) < 4:
            continue
        dx, dy = poly[1] - poly[0]
        orient = int(round(math.degrees(math.atan2(dy, dx)) / 90.0)) % 4
        # (u, v): u avanza en el sentido de lectura, v es perpendicular.
        if orient == 0:
            uv = np.stack([poly[:, 0], poly[:, 1]], axis=1)
        elif orient == 1:
            uv = np.stack([poly[:, 1], -poly[:, 0]], axis=1)
        elif orient == 2:
            uv = np.stack([-poly[:, 0], -poly[:, 1]], axis=1)
        else:
            uv = np.stack([-poly[:, 1], poly[:, 0]], axis=1)
        box = (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max())
        by_orientation.setdefault(orient, []).append((text.strip(), poly, conf, box))

    phrases: list[OcrItem] = []
    for orient, group in by_orientation.items():
        group.sort(key=lambda t: ((t[3][2] + t[3][3]) / 2, t[3][0]))
        rows: list[list] = []
        for tok in group:
            v0, v1 = tok[3][2], tok[3][3]
            for row in rows:
                r0, r1 = row[-1][3][2], row[-1][3][3]
                overlap = min(v1, r1) - max(v0, r0)
                if overlap > 0.5 * min(v1 - v0, r1 - r0):
                    row.append(tok)
                    break
            else:
                rows.append([tok])
        for row in rows:
            row.sort(key=lambda t: t[3][0])
            current = [row[0]]
            for tok in row[1:]:
                prev = current[-1]
                height = max(prev[3][3] - prev[3][2], tok[3][3] - tok[3][2])
                if tok[3][0] - prev[3][1] < 0.9 * height:
                    current.append(tok)
                else:
                    phrases.append(_phrase(current, orient))
                    current = [tok]
            phrases.append(_phrase(current, orient))
    return phrases


def _phrase(tokens: list, orient: int) -> OcrItem:
    parts = [tokens[0][0]]
    for prev, tok in zip(tokens, tokens[1:]):
        height = max(prev[3][3] - prev[3][2], tok[3][3] - tok[3][2])
        parts.append(("" if tok[3][0] - prev[3][1] < 0.12 * height else " ") + tok[0])
    pts = np.concatenate([t[1] for t in tokens])
    return OcrItem(
        text="".join(parts),
        x0=float(pts[:, 0].min()),
        y0=float(pts[:, 1].min()),
        x1=float(pts[:, 0].max()),
        y1=float(pts[:, 1].max()),
        conf=float(np.mean([t[2] for t in tokens])),
        rotation=orient * 90,
    )


class DocAIOcr:
    """Reconoce el texto de un recorte y lo devuelve en píxeles de ese recorte."""

    def __init__(self, project: str, location: str, processor_id: str, cache_dir: str | Path | None = None):
        if not processor_id:
            raise ValueError("Falta DOCAI_PROCESSOR_ID: el ID del procesador de Document AI.")
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai

        self._documentai = documentai
        self._client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        )
        self.name = self._client.processor_path(project, location, processor_id)
        self.cache = OcrCache(cache_dir) if cache_dir else None
        self.calls = 0
        self.cache_hits = 0

    def recognize(self, image: np.ndarray) -> list[OcrItem]:
        h, w = image.shape[:2]
        if h * w > MAX_PIXELS:
            raise ValueError("El recorte supera los 40 megapíxeles que admite Document AI.")
        ok, buf = cv2.imencode(".png", image)
        mime = "image/png"
        if not ok or len(buf) > MAX_BYTES:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            mime = "image/jpeg"
        payload = buf.tobytes()

        key = OcrCache.key(payload, self.name)
        if self.cache and (cached := self.cache.get(key)) is not None:
            self.cache_hits += 1
            return cached

        documentai = self._documentai
        result = self._client.process_document(
            request=documentai.ProcessRequest(
                name=self.name,
                raw_document=documentai.RawDocument(content=payload, mime_type=mime),
            )
        )
        self.calls += 1
        items = parse_document(result.document, w, h)
        if self.cache:
            self.cache.put(key, items)
        return items


def parse_document(document, width: int, height: int) -> list[OcrItem]:
    """Convierte la respuesta de Document AI en frases con su caja en píxeles."""
    tokens: list[tuple[str, np.ndarray, float]] = []
    for page in document.pages:
        for token in page.tokens:
            layout = token.layout
            text = "".join(
                document.text[int(seg.start_index) : int(seg.end_index)]
                for seg in layout.text_anchor.text_segments
            )
            verts = layout.bounding_poly.normalized_vertices
            poly = np.array([[v.x * width, v.y * height] for v in verts], dtype=float)
            tokens.append((text, poly, float(layout.confidence)))
    return group_tokens(tokens)
