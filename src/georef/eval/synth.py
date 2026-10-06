"""Mapas sintéticos con verdad conocida.

Sirven para medir el error del pipeline en píxeles y metros sin georreferenciar
nada a mano, y para probarlo sin llamar a ningún servicio externo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..coords.parse import format_dms
from ..session import OcrItem


@dataclass
class SynthSpec:
    width: int = 2400
    height: int = 1800
    margin: int = 170
    kind: str = "utm"  # utm | geo
    style: str = "lines"  # lines | ticks | crosses
    epsg: int = 32718
    origin: tuple[float, float] = (318_350.0, 8_652_700.0)  # mundo en la esquina superior izquierda del marco
    res: float = 5.0  # unidades de mundo por píxel
    step: float = 1000.0
    rotation_deg: float = 0.0
    noise: float = 5.0
    blur: float = 0.7
    clutter: int = 30
    decoys: int = 1  # trazos largos y rectos que no son cuadrícula
    vertical_side_labels: bool = True
    label_format: str = "spaced"  # plain | spaced | unit
    hemisphere_letters: bool = False  # geo: "76°45' W" en vez de "76°45'"
    legend: str = "PROYECCION UTM - ZONA 18 SUR - DATUM WGS 84 - ESCALA 1:50 000"
    seed: int = 0


@dataclass
class SynthMap:
    image: np.ndarray
    truth: np.ndarray  # 3x2: mundo = [px, py, 1] @ truth, en unidades de las etiquetas
    spec: SynthSpec
    labels: list[OcrItem] = field(default_factory=list)
    vertical_ids: set[int] = field(default_factory=set)  # índices de etiquetas rotuladas en vertical

    def pixel_to_world(self, px: np.ndarray) -> np.ndarray:
        px = np.atleast_2d(np.asarray(px, dtype=float))
        return np.column_stack([px, np.ones(len(px))]) @ self.truth


def geo_spec(**overrides) -> SynthSpec:
    """Mapa con retícula geográfica cada 5 minutos (zona de Lima)."""
    base = dict(
        kind="geo", epsg=4326, origin=(-76.9431, -11.8736), res=0.0002, step=5 / 60,
        legend="COORDENADAS GEOGRAFICAS - DATUM WGS 84 - ESCALA 1:100 000",
    )
    base.update(overrides)
    return SynthSpec(**base)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _label_text(spec: SynthSpec, value: float, axis: str) -> str:
    if spec.kind == "geo":
        text = format_dms(abs(value))
        if text.endswith("'00\""):
            text = text[:-3]
        if spec.hemisphere_letters:
            text += " " + (("W" if value < 0 else "E") if axis == "x" else ("S" if value < 0 else "N"))
        return text
    n = int(round(value))
    if spec.label_format == "plain":
        return str(n)
    spaced = f"{n:,}".replace(",", " ")
    return spaced if spec.label_format == "spaced" else f"{spaced} m{'E' if axis == 'x' else 'N'}"


def _draw_text(canvas: Image.Image, text: str, centre: tuple[float, float], font, vertical: bool = False) -> tuple[float, float, float, float]:
    """Dibuja texto centrado. Si es vertical, se lee de abajo hacia arriba. Devuelve su caja."""
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    patch = Image.new("L", (tw + 6, th + 6), 0)
    ImageDraw.Draw(patch).text((3 - left, 3 - top), text, fill=255, font=font)
    if vertical:
        patch = patch.rotate(90, expand=True)
    x0 = int(round(centre[0] - patch.width / 2))
    y0 = int(round(centre[1] - patch.height / 2))
    canvas.paste((25, 25, 25), (x0, y0), patch)
    return float(x0 + 3), float(y0 + 3), float(x0 + patch.width - 3), float(y0 + patch.height - 3)


def _clutter(img: np.ndarray, frame: tuple[int, int, int, int], spec: SynthSpec, rng: np.random.Generator) -> None:
    """Vías, curvas de nivel y trazos que no son cuadrícula."""
    fx0, fy0, fx1, fy1 = frame
    layer = img[fy0:fy1, fx0:fx1]
    h, w = layer.shape[:2]
    for _ in range(spec.clutter):
        n = int(rng.integers(5, 12))
        start = rng.uniform([0, 0], [w, h])
        steps = rng.normal(0, 1, (n, 2)).cumsum(axis=0) * rng.uniform(40, 130)
        pts = (start + steps).astype(np.int32).reshape(-1, 1, 2)
        colour = [(60, 90, 150), (90, 90, 90), (150, 110, 60)][int(rng.integers(3))]
        cv2.polylines(layer, [pts], False, colour, int(rng.integers(1, 4)), cv2.LINE_AA)
    for _ in range(spec.decoys):
        # Un tramo recto y largo, casi vertical, fuera del paso de la cuadrícula.
        x = int(rng.uniform(0.15, 0.85) * w)
        cv2.line(layer, (x, int(0.05 * h)), (x + int(rng.uniform(-25, 25)), int(0.6 * h)), (70, 70, 70), 2, cv2.LINE_AA)


def make_map(spec: SynthSpec | None = None) -> SynthMap:
    spec = spec or SynthSpec()
    rng = np.random.default_rng(spec.seed)
    w, h, mg = spec.width, spec.height, spec.margin
    fx0, fy0, fx1, fy1 = mg, mg, w - mg, h - mg
    x_org, y_org = spec.origin
    res = spec.res

    img = np.full((h, w, 3), 246, np.uint8)
    _clutter(img, (fx0, fy0, fx1, fy1), spec, rng)

    def px_of_x(value: float) -> float:
        return fx0 + (value - x_org) / res

    def py_of_y(value: float) -> float:
        return fy0 + (y_org - value) / res

    def multiples(lo: float, hi: float) -> list[float]:
        first = math.ceil(lo / spec.step - 1e-9)
        last = math.floor(hi / spec.step + 1e-9)
        return [round(k * spec.step, 9) for k in range(first, last + 1)]

    x_values = [v for v in multiples(x_org, x_org + (fx1 - fx0) * res) if fx0 + 12 < px_of_x(v) < fx1 - 12]
    y_values = [v for v in multiples(y_org - (fy1 - fy0) * res, y_org) if fy0 + 12 < py_of_y(v) < fy1 - 12]

    ink = (40, 30, 20)
    tick, arm = 20, 13

    def stroke(a: tuple[float, float], b: tuple[float, float]) -> None:
        # Coordenadas fraccionarias: la línea queda donde dice la verdad, no en el píxel entero más cercano.
        pa = (int(round(a[0] * 16)), int(round(a[1] * 16)))
        pb = (int(round(b[0] * 16)), int(round(b[1] * 16)))
        cv2.line(img, pa, pb, ink, 2, cv2.LINE_AA, shift=4)

    for v in x_values:
        x = px_of_x(v)
        if spec.style == "lines":
            stroke((x, fy0), (x, fy1))
        elif spec.style == "ticks":
            stroke((x, fy0), (x, fy0 + tick))
            stroke((x, fy1 - tick), (x, fy1))
    for v in y_values:
        y = py_of_y(v)
        if spec.style == "lines":
            stroke((fx0, y), (fx1, y))
        elif spec.style == "ticks":
            stroke((fx0, y), (fx0 + tick, y))
            stroke((fx1 - tick, y), (fx1, y))
    if spec.style == "crosses":
        for vx in x_values:
            for vy in y_values:
                x, y = px_of_x(vx), py_of_y(vy)
                stroke((x - arm, y), (x + arm, y))
                stroke((x, y - arm), (x, y + arm))
    cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (0, 0, 0), 3)

    # Etiquetas y leyenda con una fuente real (cv2 no dibuja "°").
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = _font(26)
    labels: list[OcrItem] = []
    vertical_ids: set[int] = set()

    def add(text: str, centre: tuple[float, float], vertical: bool = False) -> None:
        box = _draw_text(pil, text, centre, font, vertical)
        if vertical:
            vertical_ids.add(len(labels))
        labels.append(OcrItem(text, *box, conf=0.97, rotation=270 if vertical else 0))

    for v in x_values:
        text = _label_text(spec, v, "x")
        add(text, (px_of_x(v), fy0 - 30))
        add(text, (px_of_x(v), fy1 + 30))
    side_offset = 34 if spec.vertical_side_labels else 82
    for v in y_values:
        text = _label_text(spec, v, "y")
        add(text, (fx0 - side_offset, py_of_y(v)), spec.vertical_side_labels)
        add(text, (fx1 + side_offset, py_of_y(v)), spec.vertical_side_labels)
    if spec.legend:
        add(spec.legend, (w / 2, fy1 + 100))
    for name, pos in (("RIO CHILLON", (0.3, 0.35)), ("CERRO COLORADO", (0.62, 0.55)), ("QUEBRADA SECA", (0.45, 0.78))):
        add(name, (fx0 + pos[0] * (fx1 - fx0), fy0 + pos[1] * (fy1 - fy0)))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Verdad sin giro: X = x_org + (px - fx0)·res ; Y = y_org - (py - fy0)·res
    truth = np.array([[res, 0.0], [0.0, -res], [x_org - fx0 * res, y_org + fy0 * res]])

    if spec.rotation_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), spec.rotation_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(246, 246, 246))
        # píxel girado -> píxel original -> mundo
        back = np.vstack([cv2.invertAffineTransform(m), [0, 0, 1]])
        truth = back.T @ truth
        for i, it in enumerate(labels):
            corners = np.array([[it.x0, it.y0, 1], [it.x1, it.y0, 1], [it.x1, it.y1, 1], [it.x0, it.y1, 1]]) @ m.T
            labels[i] = OcrItem(it.text, corners[:, 0].min(), corners[:, 1].min(),
                                corners[:, 0].max(), corners[:, 1].max(), it.conf, it.rotation)

    if spec.blur:
        img = cv2.GaussianBlur(img, (0, 0), spec.blur)
    if spec.noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, spec.noise, img.shape), 0, 255).astype(np.uint8)
    return SynthMap(img, truth, spec, labels, vertical_ids)


class FakeOcr:
    """OCR simulado: devuelve las etiquetas verdaderas del mapa sintético.

    Puede perder etiquetas, alterar un dígito o exigir el recorte girado para
    leer el texto vertical, que son los fallos habituales del OCR real.
    """

    def __init__(self, synth: SynthMap, drop: float = 0.0, corrupt: float = 0.0,
                 vertical_needs_rotation: bool = False, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vertical_needs_rotation = vertical_needs_rotation
        self.items: list[tuple[OcrItem, bool]] = []
        self.calls = 0
        for i, it in enumerate(synth.labels):
            if rng.random() < drop:
                continue
            text = it.text
            digits = [k for k, ch in enumerate(text) if ch.isdigit()]
            if digits and rng.random() < corrupt:
                k = int(rng.choice(digits[:3]))
                text = text[:k] + str((int(text[k]) + int(rng.integers(1, 9))) % 10) + text[k + 1 :]
            self.items.append((OcrItem(text, it.x0, it.y0, it.x1, it.y1, it.conf, it.rotation), i in synth.vertical_ids))

    def recognize_region(self, region: tuple[int, int, int, int], rotation: int = 0) -> list[OcrItem]:
        self.calls += 1
        x, y, w, h = region
        out = []
        for it, vertical in self.items:
            if not (x <= it.cx <= x + w and y <= it.cy <= y + h):
                continue
            if vertical and self.vertical_needs_rotation and rotation != 90:
                continue
            out.append(OcrItem(it.text, it.x0, it.y0, it.x1, it.y1, it.conf, it.rotation))
        return out
