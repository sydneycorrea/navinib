"""Coordinator agent: the supervisor at the hub. Every agent reports back here, and it decides what runs next.

Briefs are written just in time, one call ahead, so no tokens go to a call that
later cancels or moves (Checkpoint 5.1). That gives three flows:

Morning plan:   Targeting (territory) -> Routing (OR-Tools) -> Compliance (providers and times) -> rep approves
Cancellation:   cancel the call -> Targeting (nearby) -> Routing (ToT) -> Compliance -> rep approves
Brief next:     when the next booked call's brief is due -> Brief (that call only) -> Compliance (claims)

A brief's timing is set per call: it is due at the call's start minus the drive there
(from the last booked call before it, or home) minus reading time, and is written
BRIEF_GEN_MARGIN_MIN before that, or right away if that time has passed. So a call
20 minutes away after a replan is briefed as soon as the rep approves, and a call 45
minutes away is briefed earlier than one next door.

Two links go both ways. If the morning route leaves a window with room for another
visit after dropping providers that didn't fit (office hours or drive time), the
Coordinator asks Targeting once for the next candidates, without the dropped ones, and
reruns Routing (at most MAX_REFILLS). A Compliance fail goes back to the agent that caused
it (Routing for times, Brief for claims), at most MAX_RETRIES times; anything still wrong
is flagged to the rep, never silently fixed. Every other handoff is one-way.

The routing is a fixed policy, so the flows are predictable and testable. The LLM
writes the plan summary for the rep; with a real model it will also read the rep's
free-text request and choose among the allowed next steps.
"""
import hashlib
import sqlite3

from navinib import guardrails
from navinib.agents.brief import partial_cause
from navinib.agents.llm import plain_summary
from navinib.config import BRIEF_GEN_MARGIN_MIN, BRIEF_READ_MIN, MAX_REFILLS, MAX_RETRIES, POLICY
from navinib.log import get_logger
from navinib.tools.drive import DriveTimes
from navinib.tools.schedule import cancel_event, hhmm, schedule_lookup, to_min

log = get_logger("navinib.agents.coordinator")

FLOWS = ("morning", "cancellation", "brief_next")   # the only things the Coordinator will do

SUMMARY_SYSTEM = """You are Navinib, a co-pilot for a pharmaceutical field rep. In at most 80 words, tell
the rep what is proposed for their day and why, mention estimated drive times, any warning and any
compliance issue that needs their review, and ask them to approve. Use only the data given."""


def coordinator(state: dict, config: dict) -> dict:
    """Decide the next step from the state; on the last step, write the result for the rep."""
    cfg = config["configurable"]
    if "schedule" not in state:
        stop = _screen_request(cfg["con"], state["request"], cfg["llm"])
        return stop or _start(cfg["con"], state["request"], cfg["drive"])
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


def brief_timing(con: sqlite3.Connection, rep_id: str, schedule: dict, event: dict, drive: DriveTimes) -> dict:
    """When a call's brief is due and when to write it: start - drive there - reading time, less a margin.

    The drive is from the last booked call that ends before this one (cancelled calls
    don't count), or from home when there is none.
    """
    before = [e for e in schedule["events"] if e["kind"] == "call" and e["status"] != "cancelled"
              and e["location_id"] and e["end_min"] <= event["start_min"] and e["event_id"] != event["event_id"]]
    origin = before[-1]["location_id"] if before else None
    drive_min = drive.between(_point(con, rep_id, origin), _point(con, rep_id, event["location_id"]))
    due = event["start_min"] - drive_min - BRIEF_READ_MIN
    return {"drive_min": drive_min, "from": origin or "home", "due_min": due, "write_min": due - BRIEF_GEN_MARGIN_MIN}


def next_due_call(con: sqlite3.Connection, rep_id: str, schedule: dict, now_min: int,
                  drive: DriveTimes) -> tuple[dict | None, dict | None]:
    """The next booked call whose brief should be written now, and the next call still waiting for its time.

    A call is due once now has reached its write time (brief_timing) and it hasn't
    started. Cancelled calls are never due, and a call with a saved brief is never due again.
    """
    briefed = {r[0] for r in con.execute("SELECT event_id FROM call_brief")}
    waiting = None
    for e in schedule["events"]:
        if e["kind"] != "call" or e["status"] == "cancelled" or e["event_id"] in briefed or e["start_min"] < now_min:
            continue
        t = brief_timing(con, rep_id, schedule, e, drive)
        if t["write_min"] > now_min:
            waiting = waiting or {"event_id": e["event_id"], "start": e["start"], **t}
            continue
        p = con.execute("SELECT full_name FROM provider WHERE provider_id = ?", (e["provider_id"],)).fetchone()
        return {"event_id": e["event_id"], "provider_id": e["provider_id"], "full_name": p["full_name"],
                "call_type": e["call_type"], "location_id": e["location_id"], "start": e["start"],
                "end": e["end"], **t}, waiting
    return None, waiting


def _point(con: sqlite3.Connection, rep_id: str, location_id: str | None) -> dict:
    """A location as a drive-time point, or the rep's home when there is none."""
    if location_id:
        loc = con.execute("SELECT * FROM location WHERE location_id = ?", (location_id,)).fetchone()
        return {"id": location_id, "lat": loc["lat"], "lon": loc["lon"]}
    rep = con.execute("SELECT * FROM rep WHERE rep_id = ?", (rep_id,)).fetchone()
    return {"id": f"home:{rep_id}", "lat": rep["home_lat"], "lon": rep["home_lon"]}


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
    """Morning plan or cancellation replan: rank, route, check, then ask for approval."""
    req = state["request"]
    refill = state.get("refill", {}).get("round", 0)
    if "targets" not in state or state["targets"]["refill"] < refill:
        why = ("next candidates for the underfilled day" if refill else
               "rank providers in the territory" if req["flow"] == "morning" else "rank providers near the freed window")
        return _go("targeting", why)
    rnd = state.get("round", 0)
    plan = state.get("plan")
    if plan is None or plan["round"] < state.get("route_round", -1) or plan["refill"] < refill:
        why = ("plan the new calls" if plan is None else f"redo the route (retry {rnd})" if
               plan["round"] < state.get("route_round", -1) else "refill the day with the next candidates")
        return _go("routing", why)
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


def _screen_request(con: sqlite3.Connection, req: dict, llm) -> dict | None:
    """Input guardrails on the request, before any agent runs. Returns a final answer to stop, or None to go on.

    A flow the Coordinator doesn't know gets a clarifying question, not a guess. A rep
    message that reports a possible adverse event or asks about an unapproved use is not
    answered: it is logged and routed out of Navinib (safety reporting, Medical Information).
    """
    text = req.get("message")
    stop = _screen_message(con, req, text, llm) if text else None
    if stop:
        return stop
    if req.get("flow") not in FLOWS:
        msg = ("I can plan your day, replan after a cancellation, or brief your next call. "
               "Which one do you need?")
        return _stop("needs_clarification", msg, [("request", "flag", f"unknown request {req.get('flow')!r}")])
    if req["flow"] == "brief_next" and not POLICY["agents"].get("brief", True):
        return _stop("plan_only", "Briefs are turned off (plan-only mode); your plan and route are unchanged.",
                     [("brief", "flag", "Brief agent turned off by the kill switch")])
    return None


def _screen_message(con: sqlite3.Connection, req: dict, text: str, llm) -> dict | None:
    """The rep's own words: an adverse event or an off-label question stops the run, whatever was asked."""
    ref = "msg-" + hashlib.sha1(f"{req['rep_id']}{req['day']}{text}".encode()).hexdigest()[:8]
    found = guardrails.screen(text, llm)
    if found["adverse_event"]:
        e = guardrails.escalate(con, "adverse_event", ref, text, req["rep_id"])
        msg = ("This may be an adverse event, so Navinib won't summarize it. It has been logged for "
               f"{e['routed_to']}; please also report it through your company's process within one business day.")
        return _stop("escalated", msg, [("rep message", "escalate", "possible adverse event")], [e])
    if guardrails.off_label_question(text):
        e = guardrails.escalate(con, "off_label", ref, text, req["rep_id"])
        msg = (f"That question is about an unapproved use, so Navinib can't answer it. Refer it to "
               f"{e['routed_to']}; the request has been logged.")
        return _stop("escalated", msg, [("rep message", "escalate", "off-label question")], [e])
    return None


def _stop(status: str, message: str, lane_items: list, escalations: list | None = None) -> dict:
    """End the run before any agent works, with a lane for the rep or the reviewers."""
    line = f"coordinator: stopped before planning ({status}): {lane_items[0][2]}"
    log.warning(line)
    return {"next": "end", "status": status, "message": message, "schedule": {}, "flags": [],
            "lanes": guardrails.lanes(lane_items), "escalations": escalations or [], "trace": [line]}


def _start(con: sqlite3.Connection, req: dict, drive: DriveTimes) -> dict:
    """First step: record a cancellation if there is one, read the schedule, and pick the first agent."""
    if req["flow"] == "cancellation":
        cancel_event(con, req["event_id"])
    schedule = schedule_lookup(con, req["rep_id"], req["day"])
    update = {"schedule": schedule, "round": 0, "status": "working"}
    if req["flow"] == "brief_next":
        due, waiting = next_due_call(con, req["rep_id"], schedule, req["now_min"], drive)
        if due is None:
            later = (f"; next brief at {hhmm(waiting['write_min'])} for the {waiting['start']} call"
                     if waiting else "")
            line = f"coordinator: brief check at {hhmm(req['now_min'])}; no call is due for a brief{later}"
            log.info(line)
            msg = "No brief due yet." + (f" The next one will be written at {hhmm(waiting['write_min'])}"
                                         f" for the {waiting['start']} call." if waiting else "")
            return {**update, "next": "end", "status": "nothing_due", "message": msg,
                    "lanes": guardrails.lanes([("brief check", "auto", "no call is due yet")]), "trace": [line]}
        update["due"] = [due]
        nxt = "brief"
        line = (f"coordinator: brief check at {hhmm(req['now_min'])}; {due['start']} {due['full_name']} is "
                f"{due['drive_min']} min from {due['from']}, brief due {hhmm(due['due_min'])} -> brief")
    else:
        nxt = "targeting"
        line = (f"coordinator: {req['flow']} request" + (f", {req['event_id']} cancelled" if req.get("event_id")
                                                        else "") + f"; {len(schedule['events'])} events today -> {nxt}")
    log.info(line)
    return {**update, "next": nxt, "trace": [line]}


def _go(agent: str, why: str) -> dict:
    """Hand the work to one agent."""
    log.info("coordinator -> %s (%s)", agent, why)
    return {"next": agent, "trace": [f"coordinator -> {agent}: {why}"]}


def _refill(plan: dict, rnd: int) -> dict:
    """Routing left room in the day after dropping providers that didn't fit: ask Targeting for the next ones."""
    update = {"refill": {"round": rnd, "exclude": plan["didnt_fit"]}}
    update.update(_go("targeting", f"day underfilled; {', '.join(plan['didnt_fit'])} didn't fit "
                                   f"(refill {rnd} of {MAX_REFILLS})"))
    return update


def _retry(state: dict, issues: list, rnd: int) -> dict:
    """Send each problem back to the agent that caused it, with what Compliance found."""
    brief_fb: dict[str, str] = {}
    for i in issues:   # every problem for a brief goes back, not only the last one
        if i["agent"] == "brief":
            brief_fb[i["provider_id"]] = "; ".join(filter(None, [brief_fb.get(i["provider_id"]), i["problem"]]))
    route_ids = sorted({i["provider_id"] for i in issues if i["agent"] == "routing"})
    fb = dict(state.get("feedback", {}))
    fb["brief"] = {**fb.get("brief", {}), **brief_fb}
    fb["exclude"] = sorted(set(fb.get("exclude", [])) | set(route_ids))
    update = {"round": rnd, "feedback": fb,
              "brief_rounds": {**state.get("brief_rounds", {}), **{pid: rnd for pid in brief_fb}}}
    if route_ids:
        update["route_round"] = rnd
    update.update(_go("routing" if route_ids else "brief",
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
        why = "the model is switched off" if "kill switch" in str(e) else (
            "the daily free-model limit is reached" if "daily" in str(e) else "the model did not answer")
        message = f"Written summary unavailable ({why}; plan built in code). " + plain_summary(data)
    items = [("ranking and route drafts", "auto", "internal work, kept in the trace"),
             ("plan", "rep_approves", "every calendar change waits for the rep (autonomy level 2)")]
    items += [(f"plan: {f['provider_id']}", "flag", f"{f['problem']} (still failing after {MAX_RETRIES} retries)")
              for f in flags]
    if plan.get("drive_estimated"):
        items.append(("drive times", "flag", "estimated, live road times unavailable"))
    if guardrails.close_options(plan):
        items.append(("replan choice", "flag", f"two best options within {POLICY['limits']['close_options_pct']}%"
                                               " of each other: the rep should choose"))
    items += [("plan", "flag", w) for w in data["warnings"]]
    line = f"coordinator: done, {len(plan['visits'])} call(s) proposed, {len(flags)} flag(s) for the rep"
    log.info(line)
    return {"next": "end", "status": "awaiting_approval", "flags": flags, "message": message,
            "lanes": guardrails.lanes(items), "escalations": [], "trace": [line]}


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
        message += f" It is partial ({partial_cause(b.get('why', ''))}), so no product claims."
    for w in state.get("compliance", {}).get("warnings", []):
        message += f" Note: {w}."
    if flags:
        message += f" Needs your review: {len(flags)} compliance issue(s) not fixed automatically."
    checks = b.get("checks", {})
    escalations = checks.get("escalations", [])
    for e in escalations:
        message += (f" A call note ({e['source_ref']}) mentions a possible adverse event: it was logged for "
                    f"{e['routed_to']} and left out of the brief. Please also report it through your company's process.")
    if checks.get("quarantined"):
        message += f" Held back for admin review: {', '.join(checks['quarantined'])} (instruction-like text)."
    if checks.get("claims_removed"):
        message += f" {len(checks['claims_removed'])} statement(s) could not be confirmed against the label and were removed."
    risky = flags or b["status"] != "ok" or checks.get("claims_removed") or state.get("compliance", {}).get("warnings")
    why = ("not confirmed by Compliance: " + "; ".join(f["problem"].removeprefix("unsupported claim: ")
                                                         for f in flags) if flags else
           "partial brief, no product claims" if b["status"] != "ok" else
           "some statements were removed, check what is left" if checks.get("claims_removed") else
           "the claims check could not run" if risky else
           "written inside limits and checked by Compliance (level 3)")
    items = [(f"brief for {v['full_name']}", "flag" if risky else "auto", why)]
    items += [(f"note {e['source_ref']}", "escalate", f"possible adverse event -> {e['routed_to']}") for e in escalations]
    items += [(f"document {r}", "escalate", "instruction-like text, quarantined for admin review")
              for r in checks.get("quarantined", [])]
    line = f"coordinator: brief for {v['event_id']} saved ({status})"
    log.info(line)
    return {"next": "end", "status": "brief_ready", "flags": flags, "message": message,
            "lanes": guardrails.lanes(items), "escalations": escalations, "trace": [line]}
