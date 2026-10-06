"""Realces para leer etiquetas tenues."""

from __future__ import annotations

import cv2
import numpy as np

MODES = ("none", "clahe", "binarize", "sharpen", "invert", "gray", "red", "green", "blue")


def apply_enhance(image: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return image
    if mode not in MODES:
        raise ValueError(f"Realce desconocido: {mode}. Opciones: {', '.join(MODES)}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        out = gray
    elif mode == "clahe":
        out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    elif mode == "binarize":
        block = max(15, (min(gray.shape) // 40) | 1)
        out = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12
        )
    elif mode == "invert":
        out = 255 - gray
    elif mode == "sharpen":
        blur = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.8, blur, -0.8, 0)
    else:
        # Un canal aislado separa tintas de color (grilla azul, curvas sepia).
        out = image[:, :, {"blue": 0, "green": 1, "red": 2}[mode]]
    return cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_GRAY2BGR)
