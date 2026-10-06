"""Índice de texto reconocido, consultable por patrón y por región."""

from __future__ import annotations

import re

from ..session import OcrItem


def _iou(a: OcrItem, b: OcrItem) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _score(item: OcrItem) -> tuple[float, int]:
    return (item.conf, len(item.text))


class OcrIndex:
    def __init__(self) -> None:
        self.items: list[OcrItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, new_items: list[OcrItem]) -> int:
        """Agrega textos; los mosaicos solapados repiten lecturas y aquí se funden."""
        added = 0
        for item in new_items:
            if not item.text.strip():
                continue
            duplicate = next((i for i, old in enumerate(self.items) if _iou(old, item) > 0.5), None)
            if duplicate is None:
                self.items.append(item)
                added += 1
            elif _score(item) > _score(self.items[duplicate]):
                self.items[duplicate] = item
        return added

    def in_region(self, x: float, y: float, w: float, h: float) -> list[OcrItem]:
        return [it for it in self.items if x <= it.cx <= x + w and y <= it.cy <= y + h]

    def search(
        self,
        pattern: str = "",
        region: tuple[float, float, float, float] | None = None,
        limit: int = 60,
    ) -> list[OcrItem]:
        items = self.in_region(*region) if region else self.items
        if pattern:
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Expresión regular inválida: {exc}") from exc
            items = [it for it in items if rx.search(it.text)]
        return sorted(items, key=lambda it: (round(it.cy / 20), it.cx))[:limit]

    def full_text(self) -> str:
        return "\n".join(it.text for it in sorted(self.items, key=lambda it: (it.cy, it.cx)))
