"""Pirámide de resoluciones para servir vistas sin reescalar la imagen completa."""

from __future__ import annotations

import cv2
import numpy as np

MAX_UPSCALE = 4.0


class Pyramid:
    def __init__(self, image: np.ndarray, min_side: int = 1200):
        self.levels = [image]
        while max(self.levels[-1].shape[:2]) > min_side * 2:
            prev = self.levels[-1]
            self.levels.append(cv2.resize(prev, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    def view(self, x: int, y: int, w: int, h: int, max_side: int) -> tuple[np.ndarray, float]:
        """Recorte (x, y, w, h) reescalado a `max_side`. Devuelve (vista, escala).

        `escala` son píxeles de la vista por píxel original.
        """
        scale = min(max_side / max(w, h), MAX_UPSCALE)
        level = 0
        while level + 1 < len(self.levels) and 0.5 ** (level + 1) >= scale:
            level += 1
        f = 0.5**level
        src = self.levels[level]
        x0, y0 = int(x * f), int(y * f)
        x1 = min(src.shape[1], max(x0 + 1, int(round((x + w) * f))))
        y1 = min(src.shape[0], max(y0 + 1, int(round((y + h) * f))))
        crop = src[y0:y1, x0:x1]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        if (out_w, out_h) == (crop.shape[1], crop.shape[0]):
            return crop.copy(), scale
        interp = cv2.INTER_AREA if out_w < crop.shape[1] else cv2.INTER_CUBIC
        return cv2.resize(crop, (out_w, out_h), interpolation=interp), scale
