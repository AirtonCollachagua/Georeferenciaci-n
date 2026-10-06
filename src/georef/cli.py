"""Línea de comandos: `georef doctor`, `georef run`, `georef eval`, `georef synth`."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import typer

from . import tracing
from .config import Settings

app = typer.Typer(add_completion=False, help="Agente de georreferenciación de imágenes de mapas.")

SUMMARY_COLUMNS = ["map", "status", "confidence", "epsg", "crs", "n_gcps", "rmse_px", "rmse_m", "fit",
                   "agent_turns", "input_tokens", "output_tokens", "cache_read_tokens", "seconds", "trace_url", "notes"]


def _settings(model: str | None = None, effort: str | None = None,
              region_hint: str | None = None, default_datum: str | None = None) -> Settings:
    settings = Settings.from_env()
    if model:
        settings.model = model
    if effort:
        settings.effort = effort
    if region_hint is not None:
        settings.region_hint = [p.strip() for p in region_hint.split(",") if p.strip()]
    if default_datum is not None:
        settings.default_datum = default_datum.strip().upper()
    tracing.configure(trace_images=settings.trace_images)
    return settings


def _ocr_provider(settings: Settings) -> Any:
    if not settings.docai_processor_id:
        return None
    from .ocr.docai import DocAIOcr

    return DocAIOcr(settings.gcp_project, settings.docai_location, settings.docai_processor_id, settings.cache_dir)


def _row(doc: dict[str, Any]) -> dict[str, Any]:
    transform = doc.get("transform") or {}
    agent = doc.get("agent") or {}
    usage = agent.get("usage") or {}
    return {
        "map": doc.get("map"), "status": doc.get("status"), "confidence": doc.get("confidence"),
        "epsg": (doc.get("crs") or {}).get("epsg"), "crs": (doc.get("crs") or {}).get("name"),
        "n_gcps": transform.get("n_gcps"), "rmse_px": transform.get("rmse_px"), "rmse_m": transform.get("rmse_m"),
        "fit": transform.get("kind"), "agent_turns": agent.get("turns"),
        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"), "seconds": doc.get("seconds"),
        "trace_url": (doc.get("trace") or {}).get("url"), "notes": " | ".join(doc.get("notes") or []),
    }


@app.command()
def run(
    path: Path = typer.Argument(..., help="Un mapa o una carpeta de mapas."),
    out: Path = typer.Option(Path("out"), help="Carpeta de salida."),
    agent: bool = typer.Option(True, "--agent/--no-agent", help="Usar el agente o solo el pre-análisis determinista."),
    workers: int = typer.Option(1, help="Mapas procesados en paralelo."),
    model: Optional[str] = typer.Option(None, help="Modelo del agente (por defecto CLAUDE_MODEL)."),
    effort: Optional[str] = typer.Option(None, help="Esfuerzo del modelo: low, medium, high, xhigh, max."),
    region_hint: Optional[str] = typer.Option(None, help='Zonas UTM candidatas, p. ej. "17S,18S,19S".'),
    default_datum: Optional[str] = typer.Option(None, help="Datum a asumir si la leyenda no lo indica."),
) -> None:
    """Georreferencia un mapa o todos los de una carpeta."""
    from .imaging.loader import SUPPORTED_SUFFIXES
    from .pipeline import process_map

    settings = _settings(model, effort, region_hint, default_datum)
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        typer.echo(f"No hay mapas en {path}.")
        raise typer.Exit(1)

    ocr = _ocr_provider(settings)
    if ocr is None:
        typer.echo("Aviso: falta DOCAI_PROCESSOR_ID; sin OCR no se leerán las etiquetas del mapa.")
    client = None
    if agent:
        from .agent.client import make_client

        client = make_client(settings)
    typer.echo(f"{len(files)} mapa(s) | agente: {settings.model if client else 'no'} | "
               f"trazas LangSmith: {'sí' if tracing.enabled() else 'no'}")

    def one(file: Path) -> dict[str, Any]:
        try:
            doc = process_map(file, out, settings, ocr, client, use_agent=agent)
        except Exception as exc:  # un mapa defectuoso no debe detener el lote
            doc = {"map": file.name, "status": "failed", "notes": [f"{type(exc).__name__}: {exc}"]}
        row = _row(doc)
        typer.echo(f"  {row['map']}: {row['status']}"
                   + (f" | EPSG:{row['epsg']}" if row["epsg"] else "")
                   + (f" | RMSE {row['rmse_px']} px" if row["rmse_px"] is not None else "")
                   + (f" | {row['notes']}" if row["notes"] else ""))
        return row

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, files))
    else:
        rows = [one(f) for f in files]

    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tracing.flush()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    typer.echo(f"Resumen: {counts} -> {out / 'summary.csv'}")


@app.command()
def doctor() -> None:
    """Comprueba el acceso a Claude en Vertex, a Document AI, a GDAL y a LangSmith."""
    settings = _settings()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn: Any) -> None:
        try:
            results.append((name, True, str(fn())))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {str(exc)[:300]}"))

    def gdal() -> str:
        import numpy as np
        import rasterio
        from rasterio.transform import Affine

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "t.tif"
            with rasterio.open(target, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8",
                               crs="EPSG:32718", transform=Affine(5, 0, 318000, 0, -5, 8652000)) as dst:
                dst.write(np.zeros((1, 4, 4), np.uint8))
            with rasterio.open(target) as src:
                return f"rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}, CRS {src.crs}"

    def claude() -> str:
        from .agent.client import make_client

        client = make_client(settings)
        reply = client.messages.create(model=settings.model, max_tokens=64, output_config={"effort": "low"},
                                       messages=[{"role": "user", "content": "Responde solo: ok"}])
        text = "".join(b.text for b in reply.content if b.type == "text").strip()
        return f"{reply.model} respondió «{text}»"

    def docai() -> str:
        if not settings.docai_processor_id:
            raise RuntimeError("Falta DOCAI_PROCESSOR_ID en .env")
        import cv2
        import numpy as np

        image = np.full((120, 520, 3), 255, np.uint8)
        cv2.putText(image, "8 650 000 N", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        items = _ocr_provider(settings).recognize(image)
        return "leyó: " + ", ".join(repr(it.text) for it in items)

    def langsmith() -> str:
        if not tracing.enabled():
            raise RuntimeError("Trazas desactivadas: definir LANGSMITH_TRACING=true y LANGSMITH_API_KEY")

        @tracing.traced("georef_doctor")
        def probe() -> dict[str, Any]:
            return tracing.current_run_info() or {}

        info = probe()
        tracing.flush()
        return f"traza de prueba enviada al proyecto {info.get('project')}: {info.get('url', info.get('run_id'))}"

    check("GDAL / rasterio", gdal)
    check(f"Claude ({settings.model})", claude)
    check("Document AI", docai)
    check("LangSmith", langsmith)
    hints = {
        "Claude": "habilitar el modelo en Vertex AI Model Garden y dar roles/aiplatform.user a la cuenta de servicio.",
        "Document AI": "crear un procesador Enterprise Document OCR, poner su ID en DOCAI_PROCESSOR_ID y dar "
                       "roles/documentai.apiUser a la cuenta de servicio.",
        "LangSmith": "poner LANGSMITH_API_KEY en .env (y LANGSMITH_ENDPOINT si la cuenta es de la región EU).",
    }
    for name, ok, detail in results:
        typer.echo(f"[{'OK' if ok else 'FALLA'}] {name}: {detail}")
        hint = next((h for key, h in hints.items() if name.startswith(key)), None)
        if not ok and hint:
            typer.echo(f"        Para resolverlo: {hint}")
    if not all(ok for _, ok, _ in results):
        raise typer.Exit(1)


@app.command("eval")
def evaluate(
    truth: Optional[Path] = typer.Argument(None, help="Carpeta con mapas y su .points de referencia. Sin ella, corre la batería sintética."),
    agent: bool = typer.Option(False, "--agent/--no-agent", help="Incluir el agente (solo con carpeta de referencia)."),
) -> None:
    """Mide el error del pipeline contra una referencia."""
    from .eval.run_eval import format_table, run_synthetic_suite, run_truth_dir

    settings = _settings()
    if truth is None:
        rows = run_synthetic_suite(settings)
    else:
        client = None
        if agent:
            from .agent.client import make_client

            client = make_client(settings)
        rows = run_truth_dir(truth, settings, _ocr_provider(settings), client)
        tracing.flush()
    typer.echo(format_table(rows))


@app.command()
def synth(out: Path = typer.Argument(Path("data/samples/sinteticos"), help="Carpeta donde escribir los mapas.")) -> None:
    """Escribe los mapas sintéticos de la batería de pruebas, con su verdad en JSON."""
    import cv2

    from .eval.run_eval import SYNTHETIC_SUITE
    from .eval.synth import make_map

    out.mkdir(parents=True, exist_ok=True)
    for name, spec, _ in SYNTHETIC_SUITE:
        synth_map = make_map(spec)
        cv2.imencode(".png", synth_map.image)[1].tofile(str(out / f"{name}.png"))
        truth = {"epsg": spec.epsg, "kind": spec.kind, "affine": synth_map.truth.tolist()}
        (out / f"{name}.truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    typer.echo(f"{len(SYNTHETIC_SUITE)} mapas en {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    app()
