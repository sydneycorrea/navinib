"""Coordinator agent: the supervisor at the hub. Every agent reports back here, and it decides what runs next.

Briefs are written just in time, one call ahead, so no tokens go to a call that
later cancels or moves (Checkpoint 5.1). That gives three flows:

Morning plan:   Targeting -> Route and Replan -> Compliance (providers and times) -> rep approves
Cancellation:   cancel the call -> Route and Replan (ToT, Targeting skipped) -> Compliance -> rep approves
Brief next:     when the prep hour before the next booked call starts -> Brief (that call only)
                -> Compliance (claims)

Two links go both ways. If the morning route leaves a window with room for another
visit after dropping providers that didn't fit (office hours or drive time), the
Coordinator asks Targeting once for the next candidates, without the dropped ones, and
reruns Route (at most MAX_REFILLS). A Compliance fail goes back to the agent that caused
it (Route for times, Brief for claims), at most MAX_RETRIES times; anything still wrong
is flagged to the rep, never silently fixed. Every other handoff is one-way.

The routing is a fixed policy, so the flows are predictable and testable. The LLM
writes the plan summary for the rep; with a real model it will also read the rep's
free-text request and choose among the allowed next steps.
"""
import sqlite3

from navinib.agents.llm import plain_summary
from navinib.config import PREP_BUFFER_MIN
from navinib.log import get_logger
from navinib.tools.schedule import cancel_event, hhmm, schedule_lookup, to_min

log = get_logger("navinib.agents.coordinator")

MAX_RETRIES = 2
MAX_REFILLS = 1

SUMMARY_SYSTEM = """You are Navinib, a co-pilot for a pharmaceutical field rep. In at most 80 words, tell
the rep what is proposed for their day and why, mention estimated drive times, any warning and any
compliance issue that needs their review, and ask them to approve. Use only the data given."""


def coordinator(state: dict, config: dict) -> dict:
    """Decide the next step from the state; on the last step, write the result for the rep."""
    cfg = config["configurable"]
    if "schedule" not in state:
        return _start(cfg["con"], state["request"])
    if state["request"]["flow"] == "brief_next":
        return _brief_flow(state, cfg)
    return _plan_flow(state, cfg)


def next_step(state: dict) -> str:
    """Edge function: the node named by the Coordinator's decision ('end' when done)."""
    return state["next"]


def briefs_to_write(state: dict) -> list[dict]:
    """Payloads for the due call when it still needs a brief (one Brief run)."""
    req, rnd = state["request"], state.get("round", 0)
    feedback = state.get("feedback", {}).get("brief", {})
    return [{"visit": v, "rep_id": req["rep_id"], "day": req["day"], "round": rnd,
             "feedback": feedback.get(v["provider_id"])} for v in _briefs_needed(state)]


def next_due_call(con: sqlite3.Connection, schedule: dict, now_min: int) -> dict | None:
    """The next booked call whose prep hour has started and that has no brief yet, or None.

    Cancelled calls are never due, so a cancelled call never gets a brief.
    """
    briefed = {r[0] for r in con.execute("SELECT event_id FROM call_brief")}
    for e in schedule["events"]:
        if (e["kind"] == "call" and e["status"] != "cancelled" and e["event_id"] not in briefed
                and e["start_min"] >= now_min and e["start_min"] - PREP_BUFFER_MIN <= now_min):
            p = con.execute("SELECT full_name FROM provider WHERE provider_id = ?", (e["provider_id"],)).fetchone()
            return {"event_id": e["event_id"], "provider_id": e["provider_id"], "full_name": p["full_name"],
                    "call_type": e["call_type"], "location_id": e["location_id"], "start": e["start"],
                    "end": e["end"]}
    return None


def approve_plan(con: sqlite3.Connection, state: dict) -> list[str]:
    """The rep approved: book the proposed calls into the schedule. Returns the new event ids."""
    req, ids = state["request"], []
    for v in state["plan"]["visits"]:
        start, end = to_min(v["start"]), to_min(v["end"])
        event_id = f"nv-{v['provider_id']}-{start}"
        con.execute("INSERT INTO schedule_event (event_id, rep_id, date, start_min, end_min, kind, provider_id,"
                    " location_id, title, call_type) VALUES (?, ?, ?, ?, ?, 'call', ?, ?, ?, ?)",
                    (event_id, req["rep_id"], req["day"], start, end, v["provider_id"], v["location_id"],
                     f"{'Call' if v['call_type'] == 'face_to_face' else 'Phone'}: {v['full_name']}",
                     v["call_type"]))
        ids.append(event_id)
    con.commit()
    log.info("approve_plan: booked %s", ids)
    return ids


def _plan_flow(state: dict, cfg: dict) -> dict:
    """Morning plan or cancellation replan: rank (morning only), route, check, then ask for approval."""
    req = state["request"]
    refill = state.get("refill", {}).get("round", 0)
    if req["flow"] == "morning" and ("targets" not in state or state["targets"]["refill"] < refill):
        return _go("targeting", "rank providers" if not refill else "next candidates for the underfilled day")
    rnd = state.get("round", 0)
    plan = state.get("plan")
    if plan is None or plan["round"] < state.get("route_round", -1) or plan["refill"] < refill:
        why = ("plan the new calls" if plan is None else f"redo the route (retry {rnd})" if
               plan["round"] < state.get("route_round", -1) else "refill the day with the next candidates")
        return _go("route_replan", why)
    if req["flow"] == "morning" and refill < MAX_REFILLS and plan["underfilled"] and plan["didnt_fit"]:
        return _refill(plan, refill + 1)
    c = state.get("compliance")
    if c is None or c["round"] < rnd:
        return _go("compliance", "check providers and times")
    if not c["passed"] and rnd < MAX_RETRIES:
        return _retry(state, c["issues"], rnd + 1)
    return _finish_plan(state, cfg["llm"], [] if c["passed"] else c["issues"])


def _brief_flow(state: dict, cfg: dict) -> dict:
    """Brief the due call, check it, and save it so it is never written twice."""
    if _briefs_needed(state):
        return _go("brief", f"write the brief for {state['due'][0]['start']} {state['due'][0]['full_name']}")
    rnd, c = state.get("round", 0), state.get("compliance")
    if c is None or c["round"] < rnd:
        return _go("compliance", "check the brief")
    if not c["passed"] and rnd < MAX_RETRIES:
        return _retry(state, c["issues"], rnd + 1)
    return _finish_brief(state, cfg["con"], [] if c["passed"] else c["issues"])


def _briefs_needed(state: dict) -> list[dict]:
    """Due calls with no brief yet in this run, or whose brief Compliance sent back this round."""
    briefs, redo = state.get("briefs", {}), state.get("brief_rounds", {})
    return [v for v in state.get("due", [])
            if v["provider_id"] not in briefs or briefs[v["provider_id"]]["round"] < redo.get(v["provider_id"], 0)]


def _start(con: sqlite3.Connection, req: dict) -> dict:
    """First step: record a cancellation if there is one, read the schedule, and pick the first agent."""
    if req["flow"] == "cancellation":
        cancel_event(con, req["event_id"])
    schedule = schedule_lookup(con, req["rep_id"], req["day"])
    update = {"schedule": schedule, "round": 0, "status": "working"}
    if req["flow"] == "brief_next":
        due = next_due_call(con, schedule, req["now_min"])
        if due is None:
            line = f"coordinator: brief check at {hhmm(req['now_min'])}; no call is due for a brief"
            log.info(line)
            return {**update, "next": "end", "status": "nothing_due", "message": "No brief due yet.",
                    "trace": [line]}
        update["due"] = [due]
        nxt = "brief"
        line = (f"coordinator: brief check at {hhmm(req['now_min'])}; prep hour started for {due['start']} "
                f"{due['full_name']} -> brief")
    else:
        nxt = "targeting" if req["flow"] == "morning" else "route_replan"
        line = (f"coordinator: {req['flow']} request" + (f", {req['event_id']} cancelled" if req.get("event_id")
                                                        else "") + f"; {len(schedule['events'])} events today -> {nxt}")
    log.info(line)
    return {**update, "next": nxt, "trace": [line]}


def _go(agent: str, why: str) -> dict:
    """Hand the work to one agent."""
    log.info("coordinator -> %s (%s)", agent, why)
    return {"next": agent, "trace": [f"coordinator -> {agent}: {why}"]}


def _refill(plan: dict, rnd: int) -> dict:
    """Route left room in the day after dropping providers that didn't fit: ask Targeting for the next ones."""
    update = {"refill": {"round": rnd, "exclude": plan["didnt_fit"]}}
    update.update(_go("targeting", f"day underfilled; {', '.join(plan['didnt_fit'])} didn't fit "
                                   f"(refill {rnd} of {MAX_REFILLS})"))
    return update


def _retry(state: dict, issues: list, rnd: int) -> dict:
    """Send each problem back to the agent that caused it, with what Compliance found."""
    brief_fb = {i["provider_id"]: i["problem"] for i in issues if i["agent"] == "brief"}
    route_ids = sorted({i["provider_id"] for i in issues if i["agent"] == "route_replan"})
    fb = dict(state.get("feedback", {}))
    fb["brief"] = {**fb.get("brief", {}), **brief_fb}
    fb["exclude"] = sorted(set(fb.get("exclude", [])) | set(route_ids))
    update = {"round": rnd, "feedback": fb,
              "brief_rounds": {**state.get("brief_rounds", {}), **{pid: rnd for pid in brief_fb}}}
    if route_ids:
        update["route_round"] = rnd
    update.update(_go("route_replan" if route_ids else "brief",
                      f"compliance failed; retry {rnd} of {MAX_RETRIES}"))
    return update


def _finish_plan(state: dict, llm, flags: list) -> dict:
    """One summary for the rep, then wait for approval. No briefs yet: they come one call ahead."""
    plan = state["plan"]
    data = {"visits": plan["visits"], "reason": plan.get("reason"), "flags": flags,
            "drive_estimated": plan.get("drive_estimated", False),
            "warnings": state.get("compliance", {}).get("warnings", [])}
    try:
        message = llm.complete("summary", SUMMARY_SYSTEM, str(data), data=data)
    except Exception as e:  # model down: the rep still gets the plan, without the written summary
        log.warning("summary unavailable: %s", e)
        message = "Written summary unavailable (model down). " + plain_summary(data)
    line = f"coordinator: done, {len(plan['visits'])} call(s) proposed, {len(flags)} flag(s) for the rep"
    log.info(line)
    return {"next": "end", "status": "awaiting_approval", "flags": flags, "message": message, "trace": [line]}


def _finish_brief(state: dict, con: sqlite3.Connection, flags: list) -> dict:
    """Save the brief (or the flagged one) for its call and tell the rep it is ready. No LLM call here."""
    v = state["due"][0]
    b = state["briefs"][v["provider_id"]]
    status = "flagged" if flags else b["status"]
    con.execute("INSERT INTO call_brief VALUES (?, ?, ?, ?, ?)",
                (v["event_id"], v["provider_id"], status, b["text"], state["request"]["now_min"]))
    con.commit()
    message = f"Brief ready for {v['start']} {v['full_name']}."
    if b["status"] == "partial":
        message += " It is partial: label and trial sources unavailable, so no product claims."
    for w in state.get("compliance", {}).get("warnings", []):
        message += f" Note: {w}."
    if flags:
        message += f" Needs your review: {len(flags)} compliance issue(s) not fixed automatically."
    line = f"coordinator: brief for {v['event_id']} saved ({status})"
    log.info(line)
    return {"next": "end", "status": "brief_ready", "flags": flags, "message": message, "trace": [line]}
