"""Trazabilidad con LangSmith.

Sin `LANGSMITH_TRACING=true` y una API key, todo este módulo es transparente:
los decoradores llaman a la función tal cual y el pipeline corre igual.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import types
from typing import Any, Callable

_state: dict[str, Any] = {"enabled": None, "client": None, "images": True, "remote": True,
                          "projects": {}, "pending_feedback": []}


def configure(enabled: bool | None = None, client: Any = None, trace_images: bool = True, remote: bool = True) -> None:
    """Fija el estado de las trazas. `enabled=None` vuelve a leerlo del entorno."""
    _state.update(enabled=enabled, client=client, images=trace_images, remote=remote)


def enabled() -> bool:
    if _state["enabled"] is not None:
        return bool(_state["enabled"])
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def get_client() -> Any:
    if _state["client"] is None:
        from langsmith import Client

        _state["client"] = Client(hide_inputs=strip_images, hide_outputs=strip_images)
    return _state["client"]


# --- Imágenes ---


def strip_images(data: Any) -> Any:
    """Sustituye las imágenes en base64 por una referencia corta.

    Cada turno del modelo reenvía toda la conversación; sin este filtro cada
    vista quedaría registrada una vez por turno.
    """
    if isinstance(data, dict):
        source = data.get("source")
        if data.get("type") == "image" and isinstance(source, dict) and source.get("type") == "base64":
            raw = str(source.get("data", ""))
            digest = hashlib.sha1(raw[:8192].encode()).hexdigest()[:10]
            return {"type": "text", "text": f"[imagen omitida: {len(raw) * 3 // 4 // 1024} KB, id {digest}]"}
        return {k: strip_images(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [strip_images(v) for v in data]
    return data


def _thumbnail(block: dict[str, Any], side: int = 768) -> str | None:
    import cv2
    import numpy as np

    try:
        raw = base64.standard_b64decode(block["source"]["data"])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        scale = min(1.0, side / max(image.shape[:2]))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.standard_b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def trace_view(blocks: Any) -> dict[str, Any]:
    """Versión para la traza del resultado de una herramienta: texto y miniaturas."""
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    try:
        view: dict[str, Any] = {"result": json.loads(text)}
    except ValueError:
        view = {"result": text}
    images = [b for b in blocks if b.get("type") == "image"]
    if images:
        if _state["images"]:
            thumbs = [_thumbnail(b) for b in images]
            view["images"] = [{"type": "image_url", "image_url": {"url": t}} for t in thumbs if t]
        else:
            view["images"] = f"{len(images)} imagen(es) no registradas (GEOREF_TRACE_IMAGES=off)"
    return view


# --- Decoradores ---


def _describe_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Las etapas reciben la sesión entera; a la traza solo va un resumen."""
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if type(value).__name__ == "MapSession":
            out["map"] = {"name": value.path.name, "size": [value.width, value.height], "sha": value.sha}
        elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
            out[key] = value
        elif type(value).__name__ == "Settings":
            out["settings"] = {"model": value.model, "effort": value.effort}
        else:
            out[key] = f"<{type(value).__name__}>"
    return out


def traced(name: str, run_type: str = "chain") -> Callable[[Callable], Callable]:
    """Registra la función como un run de LangSmith cuando las trazas están activas."""

    def decorator(fn: Callable) -> Callable:
        cache: dict[str, Any] = {"client": None, "wrapped": None}

        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not enabled():
                return fn(*args, **kwargs)
            from langsmith import traceable, tracing_context

            client = get_client()
            if cache["client"] is not client:  # el cliente puede cambiar entre configuraciones
                cache["client"] = client
                cache["wrapped"] = traceable(name=name, run_type=run_type, client=client,
                                             process_inputs=_describe_inputs)(fn)
            with tracing_context(enabled=True):
                return cache["wrapped"](*args, **kwargs)

        return inner

    return decorator


def run_tool(name: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Ejecuta una herramienta del agente como run de tipo `tool`.

    Al modelo le llega el resultado completo; a la traza, el texto y miniaturas.
    """
    if not enabled():
        return fn(**kwargs)
    holder: dict[str, Any] = {}

    def call(**kw: Any) -> dict[str, Any]:
        holder["result"] = fn(**kw)
        return trace_view(holder["result"])

    from langsmith import traceable, tracing_context

    with tracing_context(enabled=True):
        traceable(name=name, run_type="tool", client=get_client())(call)(**kwargs)
    return holder["result"]


# --- Cliente del modelo ---


def _no_completions(*_: Any, **__: Any) -> None:
    raise NotImplementedError("Este cliente no ofrece el endpoint de completions.")


def wrap_client(client: Any) -> Any:
    """Hace que cada llamada al modelo quede registrada con su uso de tokens."""
    if not enabled():
        return client
    from langsmith.wrappers import wrap_anthropic

    if not hasattr(client, "completions"):
        # wrap_anthropic da por hecho `client.completions`, que AnthropicVertex no tiene.
        client.completions = types.SimpleNamespace(create=_no_completions)
    return wrap_anthropic(client, tracing_extra={"client": get_client()})


# --- Run actual ---


def _current_run() -> Any:
    if not enabled():
        return None
    from langsmith import get_current_run_tree

    return get_current_run_tree()


def set_metadata(**metadata: Any) -> None:
    run = _current_run()
    if run is not None:
        run.add_metadata(metadata)


def current_run_info() -> dict[str, Any] | None:
    """ID y URL de la traza en curso, para enlazarla desde georef.json."""
    run = _current_run()
    if run is None:
        return None
    info = {"run_id": str(run.id), "trace_id": str(run.trace_id), "project": run.session_name}
    if _state["remote"]:
        try:
            info["url"] = run.get_url()
        except Exception as exc:  # sin red o sin permisos: la traza sigue siendo válida
            info["url_error"] = str(exc)[:200]
    return info


def _project_id(name: str) -> Any:
    """ID del proyecto de LangSmith; None si todavía no existe o no hay red."""
    if not _state["remote"]:
        return None
    if name not in _state["projects"]:
        try:
            _state["projects"][name] = get_client().read_project(project_name=name).id
        except Exception:
            return None  # el proyecto se crea al llegar la primera traza
    return _state["projects"][name]


def _post_feedback(item: dict[str, Any], session_id: Any) -> None:
    import warnings

    client = get_client()
    common = {"trace_id": item["trace_id"], "session_id": session_id, "start_time": item["start_time"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, score in item["scores"].items():
            client.create_feedback(item["run_id"], key=key, score=float(score), **common)
        client.create_feedback(item["run_id"], key="status", value=item["status"], **common)


def send_feedback(doc: dict[str, Any]) -> None:
    """Adjunta la calidad del resultado al run raíz, para filtrar mapas en LangSmith."""
    run = _current_run()
    if run is None:
        return
    transform = doc.get("transform") or {}
    scores = {
        "status_ok": 1.0 if doc.get("status") == "ok" else 0.0,
        "confidence": doc.get("confidence"),
        "rmse_px": transform.get("rmse_px"),
        "rmse_m": transform.get("rmse_m"),
        "n_gcps": transform.get("n_gcps"),
    }
    item = {"run_id": run.id, "trace_id": run.trace_id, "start_time": run.start_time,
            "project": run.session_name, "status": doc.get("status"),
            "scores": {k: v for k, v in scores.items() if v is not None}}
    try:
        session_id = run.session_id or _project_id(run.session_name)
        if session_id is None and _state["remote"]:
            # Proyecto recién creado: se envía en flush(), cuando la traza ya llegó.
            _state["pending_feedback"].append(item)
        else:
            _post_feedback(item, session_id)
    except Exception:
        pass  # las trazas nunca deben tumbar el procesamiento de un mapa


def flush() -> None:
    """Envía las trazas pendientes antes de que termine el proceso."""
    client = _state["client"]
    if client is None:
        return
    try:
        client.flush()
        pending, _state["pending_feedback"] = _state["pending_feedback"], []
        for item in pending:
            _post_feedback(item, _project_id(item["project"]))
        if pending:
            client.flush()
    except Exception:
        pass
