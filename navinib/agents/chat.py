"""Chat: the rep talks to Navinib in plain words, and each message becomes one action.

The rep's message is screened first, exactly as for any other request: a possible
adverse event or a question about an unapproved use stops there and is routed out
of Navinib. Then a small set of rules decides what was asked (rules, not the model,
so a message can never talk Navinib into an action it doesn't offer):

  plan       "plan my day"                      -> the morning plan (rep approves before booking)
  brief      "brief my next call"               -> the just-in-time brief
  cancel     "Dr. Smith cancelled", "move my 14:00"  -> replan around the freed time
  approve    "book it", "approve"               -> books the plan waiting for approval
  reject     "reject", "don't book"             -> drops it
  avoid      "what should I avoid with Okafor?" -> the do-not-raise list, from the guardrail data
  provider   "tell me about Dr. Lee"            -> an answer from that provider's own sources,
                                                   claim-checked like a brief
  help       anything else                      -> what Navinib can do

Only the provider answer uses the model, and the plan, brief and replan go through the
same agents and approval step as the buttons.
"""
import json
import re
import sqlite3

from navinib import guardrails
from navinib.agents.coordinator import _screen_message
from navinib.agents.graph import brief_next_call, run
from navinib.log import get_logger
from navinib.tools.brief import build_brief
from navinib.tools.schedule import hhmm, schedule_lookup

log = get_logger("navinib.agents.chat")

CHAT_SYSTEM = """You answer a pharmaceutical field rep's question about one healthcare provider.
Use only the facts given. Put the source id in square brackets after each fact, e.g. [N-0388].
At most 80 words, plain sentences. Never make product claims beyond the approved label text
given, and never mention anything in do_not_raise. If the facts don't answer the question, say so."""

HELP = ("I can plan your day, brief your next call, or replan when a call is cancelled "
        "(for example \"Dr. Smith cancelled\" or \"move my 14:00\"). You can also ask about a provider, "
        "such as \"what should I avoid with Dr. Okafor?\" or \"tell me about Dr. Lee\".")

_RULES = [
    ("approve", r"\b(approve|accept|book (it|them|the plan)|yes,? book|go ahead)\b"),
    ("reject", r"\b(reject|decline|don'?t book|do not book|no thanks)\b"),
    ("cancel", r"\b(cancel+ed|cancel|cancels|move|reschedul\w*|can'?t make|no[- ]show|postpone)\b"),
    ("avoid", r"\b(avoid|stay away|off[- ]limits|not (to )?(say|raise|mention|bring up)|careful)\b"),
    ("brief", r"\b(brief|prep|prepare)\b"),
    ("plan", r"\b(plan|fill)\b.*\b(day|morning|afternoon|schedule)\b|\bplan my\b|^plan\b"),
]
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)


def intent(text: str) -> str:
    """Which action the message asks for (the first rule that matches)."""
    low = text.lower()
    for name, pattern in _RULES:
        if re.search(pattern, low):
            return name
    return "provider"


def find_provider(con: sqlite3.Connection, rep_id: str, text: str) -> dict | None:
    """The rep's provider named in the message, by last name (or full name)."""
    low = text.lower()
    for p in con.execute("SELECT provider_id, full_name FROM provider WHERE rep_id = ?", (rep_id,)).fetchall():
        last = p["full_name"].split()[-1].lower()
        if re.search(rf"\b{re.escape(last)}\b", low):
            return dict(p)
    return None


def find_event(events: list, text: str, provider: dict | None) -> dict | None:
    """The booked call the message points at: by time (14:00, 2pm) or by the provider's name."""
    calls = [e for e in events if e["kind"] == "call" and e["status"] != "cancelled"]
    for m in _TIME.finditer(text):
        h, mins, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if not (m.group(2) or ampm):
            continue   # a bare number ("3 calls") is not a time
        if ampm == "pm" and h < 12:
            h += 12
        elif not ampm and 1 <= h <= 6:
            h += 12    # "move my 2:00" during the working day means 14:00
        for e in calls:
            if e["start_min"] == h * 60 + mins:
                return e
    if provider:
        for e in calls:
            if e.get("provider_id") == provider["provider_id"]:
                return e
    return None


def respond(con: sqlite3.Connection, text: str, rep_id: str, day: str, now_min: int, pending: dict | None = None,
            col=None, llm=None, drive=None, log_path=None) -> dict:
    """Answer one chat message. Returns {'intent', 'text', 'result'}: result is a full run (a plan to
    approve, a brief, an escalation) when the message started one, else None. 'approve' and 'reject'
    are returned for the app to carry out on the pending plan, so the booking path stays in one place."""
    from navinib.agents.llm import get_llm
    llm = llm or get_llm()
    req = {"rep_id": rep_id, "day": day, "now_min": now_min, "message": text}
    stop = _screen_message(con, req, text, llm)
    if stop:
        return {"intent": "escalated", "text": stop["message"], "result": {**stop, "request": req, "run_id": ""}}
    kind = intent(text)
    provider = find_provider(con, rep_id, text)
    log.info("chat: %r -> %s (provider %s)", text[:80], kind, provider and provider["provider_id"])
    common = {"col": col, "llm": llm, "drive": drive, "log_path": log_path}

    if kind in ("approve", "reject"):
        if not pending or pending.get("status") != "awaiting_approval":
            return {"intent": "help", "text": "There's no plan waiting for approval. Ask me to plan your day first.",
                    "result": None}
        return {"intent": kind, "text": "", "result": None}
    if kind == "plan":
        r = run(con, {"flow": "morning", "rep_id": rep_id, "day": day, "now_min": now_min}, **common)
        return {"intent": kind, "text": r.get("message", ""), "result": r}
    if kind == "brief":
        r = brief_next_call(con, rep_id, day, now_min, **common)
        return {"intent": kind, "text": r.get("message", ""), "result": r}
    if kind == "cancel":
        event = find_event(schedule_lookup(con, rep_id, day)["events"], text, provider)
        if event is None:
            return {"intent": "help", "result": None,
                    "text": "Which call? Give me its time (for example \"move my 14:00\") or the doctor's name."}
        r = run(con, {"flow": "cancellation", "rep_id": rep_id, "day": day, "event_id": event["event_id"],
                      "now_min": now_min}, **common)
        note = (" I can't book a call on another day yet, so add the new time in your CRM."
                if re.search(r"\b(tomorrow|next week|monday|tuesday|wednesday|thursday|friday)\b", text.lower())
                else "")
        return {"intent": kind, "result": r,
                "text": f"I've freed {hhmm(event['start_min'])} ({event['title']}).{note} {r.get('message', '')}".strip()}
    if provider is None:
        return {"intent": "help", "text": HELP, "result": None}
    if kind == "avoid":
        return {"intent": kind, "text": avoid_answer(con, col, provider, day, rep_id), "result": None}
    return {"intent": "provider", "text": provider_answer(con, col, llm, provider, text, day, rep_id), "result": None}


def avoid_answer(con: sqlite3.Connection, col, provider: dict, day: str, rep_id: str) -> str:
    """What not to raise with this provider, straight from the guardrail data (no model)."""
    b = build_brief(con, col, provider["provider_id"], day, rep_id) if col is not None else None
    items = []
    for d in (b or {}).get("do_not_raise", []):
        title = next((p["title"] for p in b["publications"] if p["pub_id"] == d["ref"]), None)
        items.append(f"{title} [{d['ref']}]" if title else f"trial {d['ref']} ({d['why']})")
    text = (f"With {provider['full_name']}, stay on the approved label. Don't raise: " + "; ".join(items) + "."
            if items else f"Nothing specific is flagged for {provider['full_name']}; stay on the approved label.")
    return text + " If they ask about any unapproved use, refer them to Medical Information."


def provider_answer(con: sqlite3.Connection, col, llm, provider: dict, question: str, day: str, rep_id: str) -> str:
    """Answer a question about one provider from their own sources, then claim-check it like a brief."""
    if col is None:
        return facts_answer(con, provider, day)
    b = build_brief(con, col, provider["provider_id"], day, rep_id, topic=question)
    notes = [n for n in b["call_notes"] if not guardrails.injection(n["text"])]
    sources = {n["ref"]: n["text"] for n in notes} | {t["ref"]: t["text"] for t in b["talking_points"]}
    facts = {"provider": b["full_name"], "tier": b["tier"], "recent_calls": b["recent_calls"],
             "lab_alerts": b["lab_alerts"], "locations_today": b["locations_today"], "call_notes": notes,
             "talking_points": b["talking_points"], "do_not_raise": b["do_not_raise"]}
    fallback = facts_answer(con, provider, day)
    try:
        text = llm.complete("chat", CHAT_SYSTEM, json.dumps({"question": question, "facts": facts}, default=str),
                            data={"fallback": fallback})
    except Exception as e:   # model down: the facts in code
        log.warning("chat answer unavailable (%s); facts only", e)
        return fallback
    text, removed = guardrails.claim_check(guardrails.redact(text)[0], sources)
    if any(d["ref"] in text for d in b["do_not_raise"]):
        return fallback + " (A model answer mentioned a do-not-raise item, so it was withheld.)"
    return text + (f" ({len(removed)} statement(s) could not be confirmed and were removed.)" if removed else "")


def facts_answer(con: sqlite3.Connection, provider: dict, day: str) -> str:
    """The provider's basic facts, built in code: tier, last calls and open lab alerts."""
    p = con.execute("SELECT full_name, tier FROM provider WHERE provider_id = ?", (provider["provider_id"],)).fetchone()
    calls = con.execute("SELECT date, call_type, outcome FROM visit WHERE provider_id = ? AND date < ?"
                        " ORDER BY date DESC LIMIT 2", (provider["provider_id"], day)).fetchall()
    last = "; ".join(f"{c['date']} {c['call_type'].replace('_', ' ')} ({c['outcome']})" for c in calls) or "none"
    return f"{p['full_name']} is tier {p['tier']}. Last calls: {last}."
