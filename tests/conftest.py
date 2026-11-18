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
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(audit, "RUNS_PATH", tmp_path / "runs.jsonl")
