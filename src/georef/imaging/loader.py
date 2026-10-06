"""Carga de mapas desde imagen o PDF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".jp2", ".webp"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
MAX_PDF_SIDE = 16000  # píxeles del lado largo al rasterizar un PDF


def load_image(path: str | Path, pdf_dpi: int = 300, page: int = 0) -> tuple[np.ndarray, float | None]:
    """Devuelve la imagen en BGR y su resolución en DPI si se conoce."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # El tamaño de página de un PDF escaneado no es fiable: no se informa DPI.
        return _load_pdf(path, pdf_dpi, page), None
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"Formato no soportado: {suffix}")

    # np.fromfile + imdecode tolera rutas con acentos en Windows.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    dpi = _read_dpi(path)
    if image is None:
        image = _load_with_pillow(path)
    return image, dpi


def _load_pdf(path: Path, dpi: int, page: int) -> np.ndarray:
    import pymupdf

    with pymupdf.open(path) as doc:
        pg = doc[page]
        long_inches = max(pg.rect.width, pg.rect.height) / 72
        images = pg.get_images(full=True)
        if images:
            # PDF escaneado: se rasteriza a la resolución de la imagen que contiene, ni más ni menos.
            native = max(max(i[2], i[3]) for i in images)
            dpi = max(1, round(native / long_inches))
        # Tope de seguridad para hojas enormes.
        dpi = max(1, min(dpi, int(MAX_PDF_SIDE / long_inches)))
        pix = pg.get_pixmap(dpi=dpi, alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if rgb.shape[2] == 1:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)


def _load_with_pillow(path: Path) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_dpi(path: Path) -> float | None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
    except Exception:
        return None
    if not dpi:
        return None
    value = float(dpi[0])
    # 72 y 96 suelen ser valores por defecto del software, no del escáner.
    return value if value > 96 else None
