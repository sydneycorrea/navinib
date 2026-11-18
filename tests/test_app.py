"""Smoke test for the Streamlit app: both views render, and the rep can plan and approve a day."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "../app/main.py"


def test_rep_view_plans_the_day_and_waits_for_approval():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Plan my day").click().run()
    assert not at.exception
    assert any("awaiting_approval" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Approve and book").click().run()
    assert not at.exception and any("Approved: booked" in c.value for c in at.caption)


def test_manager_view_is_read_only_team_level():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio[0].set_value("Regional manager").run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Tier A on goal", "Providers overdue (60+ days)"]
    assert not [b for b in at.button if b.label in ("Approve and book", "Plan my day")]
