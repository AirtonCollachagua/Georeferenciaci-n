"""Caché en disco del OCR: un mismo recorte no se paga dos veces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..session import OcrItem


class OcrCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory) / "ocr"

    @staticmethod
    def key(payload: bytes, provider: str) -> str:
        return hashlib.sha1(provider.encode() + b"|" + payload).hexdigest()

    def get(self, key: str) -> list[OcrItem] | None:
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return [OcrItem(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
        except (ValueError, TypeError):
            return None

    def put(self, key: str, items: list[OcrItem]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = [vars(it) for it in items]
        (self.dir / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
