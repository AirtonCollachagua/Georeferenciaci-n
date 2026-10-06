"""Estado de un mapa en proceso.

Todo lo pesado (imagen, índice OCR, grilla) vive aquí y no en el contexto del
modelo. Las coordenadas de píxel son siempre las de la imagen original, con el
origen arriba a la izquierda.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class OcrItem:
    """Texto reconocido con su caja en píxeles de la imagen original."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0
    rotation: int = 0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": [round(self.x0), round(self.y0), round(self.w), round(self.h)],
            "conf": round(self.conf, 2),
        }


@dataclass
class GridLine:
    """Línea de la cuadrícula: a·x + b·y + c = 0 con (a, b) unitario."""

    id: str
    axis: str  # "v" (casi vertical) o "h" (casi horizontal)
    a: float
    b: float
    c: float
    p0: tuple[float, float]
    p1: tuple[float, float]
    support: float = 1.0
    kind: str = "line"  # line | tick | cross
    on_lattice: bool = True
    value: float | None = None
    value_source: str = ""  # label | fit | agent

    def x_at(self, y: float) -> float:
        return -(self.b * y + self.c) / self.a

    def y_at(self, x: float) -> float:
        return -(self.a * x + self.c) / self.b

    def pos(self, width: int, height: int) -> float:
        """Posición de la línea medida en el centro de la imagen."""
        return self.x_at(height / 2) if self.axis == "v" else self.y_at(width / 2)

    def to_dict(self, width: int, height: int) -> dict[str, Any]:
        return {
            "id": self.id,
            "pos": round(self.pos(width, height), 1),
            "support": round(self.support, 2),
            "kind": self.kind,
            "on_lattice": self.on_lattice,
            "value": self.value,
            "value_source": self.value_source,
        }


@dataclass
class CoordLabel:
    """Etiqueta de coordenada interpretada a partir del OCR."""

    id: str
    text: str
    kind: str  # utm | geo
    value: float | None  # None si la etiqueta está abreviada
    axis: str | None  # "x" (Este / longitud), "y" (Norte / latitud) o None
    complete: bool
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0
    side: str = ""  # top | bottom | left | right | inside
    line_id: str | None = None
    status: str = "unassigned"  # unassigned | ok | outlier
    digits: str = ""

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "value": self.value,
            "axis": self.axis,
            "complete": self.complete,
            "side": self.side,
            "center": [round(self.cx), round(self.cy)],
            "line_id": self.line_id,
            "status": self.status,
        }


@dataclass
class GCP:
    """Punto de control: píxel ↔ coordenada en las unidades de las etiquetas."""

    id: str
    px: float
    py: float
    x: float
    y: float
    source: str = "grid"  # grid | agent
    enabled: bool = True
    residual_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["px"] = round(self.px, 2)
        d["py"] = round(self.py, 2)
        if self.residual_px is not None:
            d["residual_px"] = round(self.residual_px, 3)
        return d


@dataclass
class AxisFit:
    """Relación valor ↔ posición de una familia de líneas."""

    axis: str  # x | y
    kind: str  # utm | geo
    step: float
    slope: float
    intercept: float
    n_labels: int
    n_inliers: int
    sign: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Frame:
    """Marco del mapa como caja alineada a los ejes."""

    x0: float
    y0: float
    x1: float
    y1: float
    source: str = "grid"

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x0 - margin <= x <= self.x1 + margin) and (
            self.y0 - margin <= y <= self.y1 + margin
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x0),
            "y": round(self.y0),
            "w": round(self.x1 - self.x0),
            "h": round(self.y1 - self.y0),
            "source": self.source,
        }


class MapSession:
    """Un mapa y todo lo que se sabe de él."""

    def __init__(self, image: np.ndarray, path: str | Path = "", dpi: float | None = None):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = image.shape[:2]
        self.path = Path(path) if path else Path("mapa")
        self.dpi = dpi
        self.sha = hashlib.sha1(np.ascontiguousarray(image[::8, ::8]).tobytes()).hexdigest()[:16]

        from .imaging.annotate import RefGrid
        from .imaging.pyramid import Pyramid
        from .ocr.index import OcrIndex

        self.pyramid = Pyramid(image)
        self.ref_grid = RefGrid.for_image(self.width, self.height)
        self.ocr = OcrIndex()
        self.ocr_provider: Any = None

        self.frame: Frame | None = None
        self.lines: list[GridLine] = []
        self.grid_info: dict[str, Any] = {}
        self.labels: list[CoordLabel] = []
        self.axis_fits: dict[str, AxisFit] = {}
        self.label_kind: str = ""  # utm | geo

        self.crs_epsg: int | None = None
        self.crs_evidence: list[str] = []
        self.crs_confident: bool = False

        self.gcps: list[GCP] = []
        self.fit: Any = None  # georef.transform.FitResult
        self.result: dict[str, Any] | None = None
        self.warnings: list[str] = []

    # --- utilidades ---

    @property
    def name(self) -> str:
        return self.path.stem

    def line(self, line_id: str) -> GridLine | None:
        wanted = line_id.strip().upper()
        return next((ln for ln in self.lines if ln.id == wanted), None)

    def lines_of(self, axis: str) -> list[GridLine]:
        return [ln for ln in self.lines if ln.axis == axis]

    def clamp_region(self, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        x0 = int(max(0, min(self.width - 1, math.floor(x))))
        y0 = int(max(0, min(self.height - 1, math.floor(y))))
        x1 = int(max(x0 + 1, min(self.width, math.ceil(x + w))))
        y1 = int(max(y0 + 1, min(self.height, math.ceil(y + h))))
        return x0, y0, x1 - x0, y1 - y0

    def parse_region(self, region: str) -> tuple[int, int, int, int]:
        """Acepta "" (todo), "x,y,w,h" en píxeles o celdas de la rejilla ("C4", "B2:C3")."""
        text = (region or "").strip()
        if not text or text.lower() in {"all", "todo", "full"}:
            return 0, 0, self.width, self.height
        if "," in text:
            parts = [float(p) for p in text.replace(";", ",").split(",")]
            if len(parts) != 4:
                raise ValueError('La región en píxeles debe ser "x,y,w,h".')
            return self.clamp_region(*parts)
        return self.clamp_region(*self.ref_grid.parse(text))
