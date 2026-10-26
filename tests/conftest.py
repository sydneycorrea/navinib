import pytest

from navinib.data.generate import generate
from navinib.db import connect


@pytest.fixture()
def con(tmp_path):
    path = generate(tmp_path / "test.db")
    c = connect(path)
    yield c
    c.close()
