"""El bucle del agente y sus trazas, contra un modelo y un LangSmith simulados.

No se llama a ningún servicio: el transporte HTTP devuelve turnos guionizados
y la sesión de LangSmith solo anota lo que se le envía.
"""

import json
import re

import httpx2
import pytest
import requests
from anthropic import AnthropicVertex
from langsmith import Client

from georef import tracing
from georef.config import Settings
from georef.eval.synth import FakeOcr, SynthSpec, make_map
from georef.pipeline import process_session
from georef.session import MapSession


def tool_use(i, name, **inputs):
    return {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": inputs}


class FakeModel:
    """Responde cada petición con el siguiente turno del guion y guarda lo recibido."""

    def __init__(self, script, status=200):
        self.script, self.status, self.requests = script, status, []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append((str(request.url), json.loads(request.content)))
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"type": "not_found_error", "message": "modelo no habilitado"}})
        turn = len(self.requests) - 1
        content = self.script[min(turn, len(self.script) - 1)]
        stop = "refusal" if content == "refusal" else "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return httpx2.Response(200, json={
            "id": f"msg_{turn}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [] if content == "refusal" else content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 50, "cache_read_input_tokens": 800 if turn else 0,
                      "cache_creation_input_tokens": 200},
        })

    def client(self):
        return AnthropicVertex(project_id="p", region="global", access_token="t", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(self)))


class FakeLangSmith(requests.Session):
    def __init__(self):
        super().__init__()
        self.calls = []

    def request(self, method, url, **kwargs):
        data = kwargs.get("data")
        self.calls.append((method, url, json.loads(data) if data else None))
        response = requests.Response()
        response.status_code, response._content, response.url = 200, b"{}", url
        return response

    def runs(self):
        return [p for m, u, p in self.calls if m == "POST" and u.endswith("/runs")]


@pytest.fixture
def session():
    synth = make_map(SynthSpec(rotation_deg=0.8, seed=3))
    s = MapSession(synth.image, "demo.png")
    s.ocr_provider = FakeOcr(synth)
    return s


GOOD_SCRIPT = [
    [{"type": "thinking", "thinking": "Compruebo la grilla en dos esquinas.", "signature": "s"},
     tool_use(1, "view_image", region="A1:B2", max_side=1024), tool_use(2, "view_grid")],
    [tool_use(3, "validate")],
    [tool_use(4, "submit_result", status="ok", epsg=32718, confidence=0.93, notes="Grilla y leyenda comprobadas.")],
]


def test_agent_verifies_and_submits(session, tmp_path):
    model = FakeModel(GOOD_SCRIPT)
    doc = process_session(session, tmp_path, Settings(), client=model.client(), use_agent=True)

    assert doc["status"] == "ok" and doc["confidence"] == 0.93
    assert doc["agent"]["turns"] == 3 and doc["agent"]["submitted"]
    assert doc["agent"]["usage"] == {"input_tokens": 3000, "output_tokens": 150,
                                     "cache_read_input_tokens": 1600, "cache_creation_input_tokens": 600}

    url, first = model.requests[0]
    assert "publishers/anthropic/models/claude-opus-5-5" in url
    assert first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert first["output_config"] == {"effort": "high"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert len(first["tools"]) == 14 and "tool_choice" not in first
    assert [c["type"] for c in first["messages"][0]["content"]] == ["image", "text"]

    # Las dos herramientas del primer turno vuelven juntas, cada una con su imagen.
    results = model.requests[1][1]["messages"][-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert all([b["type"] for b in r["content"]] == ["text", "image"] for r in results)

    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["name"] for e in events if e["type"] == "tool_use"] == ["view_image", "view_grid", "validate", "submit_result"]
    assert events[0]["type"] == "thinking" and events[-1]["type"] == "end"
    assert "base64" not in (tmp_path / "trace.jsonl").read_text(encoding="utf-8")


def test_checks_overrule_an_ok_the_evidence_does_not_support(tmp_path):
    """El agente dice "ok", pero el mapa no declara zona ni datum: queda para revisión."""
    synth = make_map(SynthSpec(legend="CARTA TOPOGRAFICA", seed=5))
    s = MapSession(synth.image, "sin_leyenda.png")
    s.ocr_provider = FakeOcr(synth)
    model = FakeModel([[tool_use(1, "submit_result", status="ok", epsg=0, confidence=0.9, notes="Se ve bien.")]])
    doc = process_session(s, tmp_path, Settings(), client=model.client(), use_agent=True)
    assert doc["status"] == "needs_review"
    assert any("Rebajado a revisión" in n for n in doc["notes"])


def test_agent_can_declare_no_coordinates(tmp_path):
    synth = make_map(SynthSpec(style="none", seed=13))
    s = MapSession(synth.image, "croquis.png")
    s.ocr_provider = FakeOcr(synth, drop=1.0)
    model = FakeModel([[tool_use(1, "submit_result", status="no_coordinates", epsg=0, confidence=0.95,
                                 notes="Sin cuadrícula ni etiquetas.")]])
    doc = process_session(s, tmp_path, Settings(), client=model.client(), use_agent=True)
    assert doc["status"] == "no_coordinates" and "transform" not in doc


def test_model_not_enabled_falls_back_to_the_prepass(session, tmp_path):
    """Si Vertex responde 404, el mapa se procesa igual con el resultado determinista."""
    doc = process_session(session, tmp_path, Settings(), client=FakeModel([], status=404).client(), use_agent=True)
    assert "no está disponible" in doc["agent"]["error"]
    assert doc["status"] == "ok" and doc["transform"]["rmse_px"] < 0.5
    assert any("no pudo ejecutarse" in w for w in doc["warnings"])


def test_turn_limit_and_refusal_do_not_pass_as_success(session, tmp_path):
    endless = FakeModel([[tool_use(1, "validate")]])
    doc = process_session(session, tmp_path, Settings(max_turns=3), client=endless.client(), use_agent=True)
    assert doc["agent"]["turns"] == 3 and not doc["agent"]["submitted"]
    assert any("sin llamar a submit_result" in w for w in doc["warnings"])

    synth = make_map(SynthSpec(seed=3))
    other = MapSession(synth.image, "otro.png")
    other.ocr_provider = FakeOcr(synth)
    refused = process_session(other, tmp_path / "r", Settings(), client=FakeModel(["refusal"]).client(), use_agent=True)
    assert refused["agent"]["stop_reason"] == "refusal" and refused["status"] != "ok"


def test_langsmith_trace_tree_without_full_images(session, tmp_path):
    fake = FakeLangSmith()
    ls = Client(api_url="http://langsmith.test", api_key="k", session=fake, auto_batch_tracing=False,
                hide_inputs=tracing.strip_images, hide_outputs=tracing.strip_images)
    tracing.configure(enabled=True, client=ls, remote=False)
    model = FakeModel(GOOD_SCRIPT)
    doc = process_session(session, tmp_path, Settings(), client=tracing.wrap_client(model.client()), use_agent=True)

    runs = {r["id"]: r for r in fake.runs()}
    by_name: dict[str, list] = {}
    for r in runs.values():
        by_name.setdefault(r["name"], []).append(r)

    def parent(run):
        return runs[run["parent_run_id"]]["name"] if run.get("parent_run_id") else None

    root = by_name["georef_map"][0]
    assert parent(root) is None and doc["trace"]["run_id"] == root["id"]
    assert {parent(by_name[n][0]) for n in ("prepass", "agent", "export")} == {"georef_map"}
    assert {parent(by_name[n][0]) for n in ("detect_grid", "ocr_tiles", "auto_fit")} == {"prepass"}

    # Un run por turno del modelo y uno por herramienta, todos bajo "agent".
    assert len(by_name["ChatAnthropic"]) == 3 and all(r["run_type"] == "llm" for r in by_name["ChatAnthropic"])
    for name in ("view_image", "view_grid", "validate", "submit_result"):
        assert by_name[name][0]["run_type"] == "tool" and parent(by_name[name][0]) == "agent"

    # Ningún turno del modelo arrastra las imágenes en base64.
    for run in by_name["ChatAnthropic"]:
        assert len(json.dumps(run["inputs"])) < 60_000
        assert "imagen omitida" in json.dumps(run["inputs"], ensure_ascii=False)
    everything = json.dumps([p for _, _, p in fake.calls if p is not None])
    long_base64 = re.findall(r"[A-Za-z0-9+/]{20000,}", everything)
    assert len(long_base64) == everything.count("data:image/jpeg;base64,") == 2  # solo las dos miniaturas

    feedback = {p["key"]: p for m, u, p in fake.calls if m == "POST" and u.endswith("/feedback")}
    assert feedback["status"]["value"] == "ok" and feedback["status_ok"]["score"] == 1.0
    assert feedback["rmse_px"]["score"] < 0.5 and feedback["rmse_px"]["run_id"] == root["id"]


def test_trace_images_can_be_turned_off(session, tmp_path):
    fake = FakeLangSmith()
    ls = Client(api_url="http://langsmith.test", api_key="k", session=fake, auto_batch_tracing=False,
                hide_inputs=tracing.strip_images, hide_outputs=tracing.strip_images)
    tracing.configure(enabled=True, client=ls, remote=False, trace_images=False)
    model = FakeModel(GOOD_SCRIPT)
    process_session(session, tmp_path, Settings(), client=tracing.wrap_client(model.client()), use_agent=True)
    everything = json.dumps([p for _, _, p in fake.calls if p is not None], ensure_ascii=False)
    assert "data:image" not in everything and not re.findall(r"[A-Za-z0-9+/]{20000,}", everything)
    assert "no registradas" in everything
