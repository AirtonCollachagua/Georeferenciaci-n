"""Error de una georreferenciación frente a la verdad."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ..coords.crs import world_coords
from ..georef.transform import to_metres

if TYPE_CHECKING:
    from ..session import MapSession


def error_vs_truth(session: "MapSession", truth: Callable[[np.ndarray], np.ndarray], samples: int = 9) -> dict[str, float]:
    """Compara la transformación ajustada con la verdadera en una malla de puntos.

    `truth` lleva píxeles a coordenadas en las unidades de las etiquetas del mapa.
    """
    fit = session.fit
    if fit is None:
        raise ValueError("La sesión no tiene transformación ajustada.")
    xs = np.linspace(0, session.width - 1, samples)
    ys = np.linspace(0, session.height - 1, samples)
    px = np.array([(x, y) for y in ys for x in xs])
    expected = truth(px)
    if fit.epsg:
        expected = world_coords(session, fit.epsg, expected)
    metres = to_metres(fit.pixel_to_world(px) - expected, expected, fit.units == "deg")
    pixel = np.hypot(*(fit.world_to_pixel(expected) - px).T)
    return {
        "rmse_px": float(np.sqrt(np.mean(pixel**2))),
        "max_px": float(pixel.max()),
        "rmse_m": float(np.sqrt(np.mean(metres**2))),
        "max_m": float(metres.max()),
    }
