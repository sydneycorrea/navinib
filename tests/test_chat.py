"""The chat: each message is screened, then mapped to one action by rules."""
import pytest

from navinib import config
from navinib.agents.chat import find_event, intent, respond
from navinib.agents.llm import StubLLM
from navinib.tools.schedule import schedule_lookup

DAY, REP = config.DEMO_DATE, "rep-01"


@pytest.mark.parametrize("text,kind", [
    ("Plan my day", "plan"), ("can you plan my morning?", "plan"), ("Brief my next call", "brief"),
    ("Dr. Smith cancelled", "cancel"), ("move my 14:00 to tomorrow", "cancel"),
    ("what should I avoid saying to Dr. Okafor?", "avoid"), ("book it", "approve"), ("reject", "reject"),
    ("tell me about Dr. Lee", "provider"),
])
def test_intent(text, kind):
    assert intent(text) == kind


def test_find_event_by_time_and_name(con):
    events = schedule_lookup(con, REP, DAY)["events"]
    smith = next(e for e in events if "Smith" in e["title"])
    assert find_event(events, "move my 14:00", None)["event_id"] == smith["event_id"]
    assert find_event(events, "move my 2pm", None)["event_id"] == smith["event_id"]
    assert find_event(events, "I have 3 calls", None) is None


def test_chat_cancel_replans_and_waits_for_approval(con):
    r = respond(con, "move my 14:00 to tomorrow", REP, DAY, 8 * 60 + 20, llm=StubLLM())
    assert r["intent"] == "cancel" and r["result"]["status"] == "awaiting_approval"   # nothing booked yet
    assert r["text"].startswith("I've freed 14:00") and "CRM" in r["text"]


def test_chat_off_label_question_is_escalated_not_answered(con):
    r = respond(con, "Can Product X be used first-line for Dr. Okafor?", REP, DAY, 500, llm=StubLLM())
    assert r["intent"] == "escalated" and "Medical Information" in r["text"]


def test_chat_approve_needs_a_pending_plan(con):
    assert "no plan waiting" in respond(con, "book it", REP, DAY, 500, llm=StubLLM())["text"]


def test_chat_provider_question_without_index_uses_facts(con):
    r = respond(con, "tell me about Dr. Okafor", REP, DAY, 500, llm=StubLLM())
    assert r["intent"] == "provider" and "tier" in r["text"]


def test_chat_unknown_message_gets_help(con):
    assert "plan your day" in respond(con, "hello", REP, DAY, 500, llm=StubLLM())["text"]
