import pytest

from georef import tracing
from georef.config import Settings
from georef.eval.synth import FakeOcr, SynthSpec, make_map
from georef.pipeline import run_prepass
from georef.session import MapSession


@pytest.fixture(autouse=True)
def _tracing_off():
    """Las pruebas no envían trazas salvo que una lo pida de forma explícita."""
    tracing.configure(enabled=False)
    yield
    tracing.configure(enabled=False)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def utm_map():
    """Mapa UTM sintético, girado 0,8°, ya pasado por el pre-análisis."""
    synth = make_map(SynthSpec(rotation_deg=0.8, seed=3))
    session = MapSession(synth.image, "utm_demo.png")
    session.ocr_provider = FakeOcr(synth)
    summary = run_prepass(session, Settings())
    return synth, session, summary
