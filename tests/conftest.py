import pytest

from navinib.data.generate import generate
from navinib.db import connect


@pytest.fixture()
def con(tmp_path):
    """A freshly generated test database for each test, so tests can't affect each other."""
    path = generate(tmp_path / "test.db")
    c = connect(path)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Tests never call a real model and never write to the real run log."""
    from navinib import audit
    from navinib.agents import llm
    monkeypatch.setattr(llm, "_dotenv", lambda: {})
    for var in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "NAVINIB_MODEL", "NAVINIB_GROQ_MODEL", "NAVINIB_ALLOW_PAID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(audit, "RUNS_PATH", tmp_path / "runs.jsonl")
