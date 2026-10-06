"""Agente de georreferenciación de imágenes de mapas."""

import os

# Los mapas escaneados superan con facilidad el límite por defecto de OpenCV.
# Debe fijarse antes de importar cv2.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40))

__version__ = "0.1.0"
