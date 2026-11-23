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
    assert not at.exception and any("Booked" in m.value for m in at.markdown)
    assert at.session_state["chat"][-1]["text"].startswith("Booked")


def test_chat_box_answers_and_shows_the_conversation():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("what should I avoid with Dr. Okafor?").run()
    assert not at.exception
    turns = at.session_state["chat"]
    assert turns[-2] == {"role": "user", "text": "what should I avoid with Dr. Okafor?"}
    assert "Medical Information" in turns[-1]["text"]
    at.chat_input[0].set_value("plan my day").run()
    assert any("awaiting_approval" in m.value for m in at.markdown)
    at.chat_input[0].set_value("book it").run()
    assert at.session_state["chat"][-1]["text"].startswith("Booked")


def test_manager_view_is_read_only_team_level():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio[0].set_value("Regional manager").run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Tier A on goal", "Providers overdue (60+ days)"]
    assert not [b for b in at.button if b.label in ("Approve and book", "Plan my day")]


def test_cancelled_call_shows_until_the_replan_is_booked():
    at = AppTest.from_file(APP, default_timeout=120).run()
    struck = lambda: any("Call: Dr. Helen Smith" in m.value and "line-through" in m.value  # noqa: E731
                         for m in at.markdown)
    [s for s in at.selectbox if s.label == "Time now (simulated)"][0].set_value("13:05").run()
    at.chat_input[0].set_value("Dr. Smith cancelled").run()
    assert struck()          # while the replan waits for approval, the cancelled call stays in view
    at.chat_input[0].set_value("book it").run()
    assert not struck()      # once new calls are booked in its time, it is gone
